# Import required libraries
from glob import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import holoviews as hv

hv.extension('bokeh')

# 1. Data wrangling functions

def list_lines(file_name):
    """Returns a list of integer lists."""
    with open(file_name) as file:
        lines = file.read().splitlines()
    line_list = [[int(n) for n in ll.split()] for ll in lines]
    return line_list


def set_params(line_list):
    top_line = line_list[0]
    params = {'DRONE_COUNT': top_line[2],
              'WT_CAP': top_line[4],
              'END_TIME': top_line[3],
              }
    return params


def find_wh_lines(line_list):
    """Provides the dividing line between warehouse and
    order sections in the line list."""
    wh_count = line_list[3][0]
    wh_endline = (wh_count*2)+4
    return wh_endline


def get_weights(line_list):
    weights = np.array(line_list[2])
    return weights.astype(np.int16)


def get_inventories(line_list):
    """Returns a 2-d array of P products by W warehouses."""
    wh_endline = find_wh_lines(line_list)
    invs = line_list[5:wh_endline+1:2]
    supply = np.array(invs).transpose()
    return supply.astype(np.int16)


def get_orders(line_list):
    """Returns a 2-d array of P products by C orders."""
    wh_endline = find_wh_lines(line_list)
    demand = np.zeros((line_list[1][0], line_list[wh_endline][0]),
                            dtype=np.int16)
    orders = line_list[wh_endline+3::3]
    for i,ord in enumerate(orders):
        for prod in ord:
            demand[prod, i] += 1
    return demand.astype(np.int16)


def get_locs(line_list):
    wh_endline = find_wh_lines(line_list)
    wh_locs = np.array(line_list[4:wh_endline:2])
    cust_locs = np.array(line_list[wh_endline+1::3])
    return wh_locs.astype(np.int16), cust_locs.astype(np.int16)

# 2. Main function to load the data
files = ['/Users/ozgurmertkelleci/Desktop/drone_delivery/busy_day.in']
line_list = list_lines(files[0])

params = set_params(line_list)
supply = get_inventories(line_list)
demand = get_orders(line_list)
wh_locs, cust_locs = get_locs(line_list)
weights = get_weights(line_list)

print(params)
for array in ['supply', 'wh_locs', 'demand', 'cust_locs', 'weights']:
    print(array, eval(array).shape)

# 3. Visualization
freqs, edges = np.histogram(weights, 20)
wt_prod = hv.Histogram((edges, freqs)).options(xlabel="product weights", width=250, title='Weight Distributions')

order_weights = (weights.reshape(weights.size, -1)* demand).sum(axis=0)
freqs, edges = np.histogram(order_weights, 20)
wt_orders = hv.Histogram((edges, freqs)).options(xlabel="order weights", width=400)

surplus = hv.Curve(supply.sum(axis=1) - demand.sum(axis=1)).options(width=500, xlabel='Product Number', ylabel='Surplus', title='Total Surplus')

customers = hv.Points(np.fliplr(cust_locs)).options(width=600, height=400, title='Warehouse and Customer Locations')
warehouses = hv.Points(np.fliplr(wh_locs)).options(size=8, alpha=0.5)

# Save the visualization to an HTML file
hv.save(hv.Layout(wt_prod + wt_orders).options(shared_axes=False) * surplus * (customers * warehouses), 'visualization.html')

# Notify the user where the output is saved
print("Visualization saved as 'visualization.html'. Open it in a browser.")

# 4. Optimization Routine 1: Warehouse assignments
def assign_whs(supply, wh_locs, demand, cust_locs):
    assignments = []
    count = 0
    distances = distance_matrix(cust_locs, wh_locs)
    
    for i in range(400):  # iterate over products
        item_count = 0

        # Initialize solver (linear programming)
        solver = pywraplp.Solver.CreateSolver('GLOP')

        if not solver:
            raise ValueError('Solver not created.')

        # Variables: x[warehouse, customer] = number of products to transport
        x = {}
        for w in range(10):
            for c in range(1250):
                x[(w, c)] = solver.IntVar(0.0, solver.infinity(), f'x_{w}_{c}')

        # Constraints: Each product should be delivered by a warehouse to a customer
        for c in range(1250):
            solver.Add(solver.Sum(x[(w, c)] for w in range(10)) == demand[i, c])

        # Objective: Minimize the total cost (product of distance and quantity)
        objective = solver.Objective()
        for w in range(10):
            for c in range(1250):
                objective.SetCoefficient(x[(w, c)], distances[c][w] * demand[i, c])
        objective.SetMinimization()

        # Solve the problem
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            for w in range(10):
                for c in range(1250):
                    quantity = x[(w, c)].solution_value()
                    if quantity > 0:
                        assignments.append([w, c, i, int(quantity), distances[c][w]])
                        item_count += int(quantity)
        count += item_count

    print(f"Products available: {supply.sum()} \n"
          f"Products ordered: {demand.sum()} \n"
          f"Products delivered: {count}")              
    return np.array(assignments)

assignments = assign_whs(supply, wh_locs, demand, cust_locs)

def set_loads(assignments):
    # Convert assignments into a DataFrame
    assign_df = pd.DataFrame(assignments, columns=['wh', 'cust', 'prod_', 'quant', 'dist'])
    
    # Ensure prod_ is treated as an integer when indexing weights
    assign_df['prod_'] = assign_df['prod_'].astype(int)
    
    # Example of processing the load for each warehouse and customer
    assign_df['load_weight'] = assign_df['quant'] * weights[assign_df['prod_'].to_numpy()]
    assign_df['load_tag'] = assign_df['load_weight'].eq(assign_df['load_weight']).cumsum() - 1
    
    # Assign drone_id (you can define your method of assigning drones here)
    assign_df['drone_id'] = assign_df.groupby('wh').cumcount()

    # Further processing can be added here for managing load balance, etc.
    return assign_df


# Now call the set_loads function
assign_df = set_loads(assignments)
assign_df

# 6. Final DataFrame adjustments and output

def write_instrux(df_load, df_deliver, sub):
    line_count_load = 0
    for tup in df_load.itertuples():
        wh_text = f"{tup.drone_id} L {tup.wh} {tup.prod_} {tup.quant}\n"
        sub.write(wh_text)
        line_count_load +=1
    for tup in df_deliver.itertuples():
        cust_text = f"{tup.drone_id} D {tup.cust} {tup.prod_} {tup.quant}\n"
        sub.write(cust_text)
        line_count_load +=1
    return line_count_load, wh_text, cust_text

with open('submission.csv', 'w') as sub:
    sub.write(f"{len(assign_df)}\n")
    line_count = 0    
    drone_tag = assign_df.drop_duplicates(['drone_id', 'load_tag'])

    for dt in drone_tag.itertuples():
        df_load_tag = assign_df[(assign_df.load_tag == dt.load_tag) & \
                                          (assign_df.drone_id == dt.drone_id)]
        df_deliver_tag = assign_df[(assign_df.load_tag == dt.load_tag) & \
                                          (assign_df.drone_id == dt.drone_id)]
    
        line_count_load, wh_text, cust_text = write_instrux(df_load_tag, df_deliver_tag, sub)
        line_count += line_count_load

print('Sample output: \n', wh_text, cust_text)
print('Line check:', len(assign_df), line_count)
