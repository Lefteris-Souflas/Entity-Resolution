# Import necessary Libraries
# Task A
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pprint import pprint
import html
from contextlib import redirect_stdout
# Task B
from math import comb
# Task C
import sqlite3
from itertools import combinations
from tqdm.auto import tqdm
nltk.download('stopwords')


# TASK A ###############################################################################################################
# Define the function token_blocking that takes a DataFrame as input and returns an inverted index
def token_blocking(dataframe):
    # Create an empty dictionary to store the blocking index
    index = {}
    # A list of English stopwords from nltk.corpus
    en_stop = stopwords.words('english')
    # Iterate over DataFrame's rows as Pandas named tuples without returning the index as the first element of the tuple
    for row in dataframe.itertuples(index=False):
        # Extract the ID and all other attribute values from the row tuple
        id_, *attributes = row
        # Iterate over each attribute value
        for attr in attributes:
            # Check if the attribute value is indeed available (not null)
            if pd.notna(attr):
                # Transform the attribute value to lowercase and split into individual words
                tokens = str(attr).lower().split()
                # For each token in the attribute's value add the ID to the corresponding blocking key's value set
                for token in tokens:
                    # Filter out stopwords
                    if token not in en_stop:
                        # If the token is not in the index, create a new entry with the ID as the value set
                        if token not in index:
                            index[token] = {id_}
                        # If the token already exists in the index, add the ID to its value set
                        else:
                            index[token].add(id_)
    # Create a list to store keys to delete
    keys_to_delete = []
    # Iterate over each value (set of IDs) in the blocking index
    for key, value in index.items():
        # Each block should contain at least two entities, otherwise add the key to the keys_to_delete list
        if len(value) < 2:
            keys_to_delete.append(key)
    # Delete the keys along with their values from the dictionary
    for key in keys_to_delete:
        del index[key]
    # Return the blocking index as a dictionary
    return index


# Function to decode HTML special characters in a cell
def decode_html_special_chars(text):
    return html.unescape(text) if isinstance(text, str) else text


# Load the data from ER-Data.csv
print("START")
print("Loading data...")
data_path = "../Data/ER-Data.csv"
df = pd.read_csv(data_path, sep=";")
print("Data loaded successfully!\n")

# Remove HTML special characters, like `&amp;`, `&hellip;`, `&lt;`, `&quot;`
print("Removing HTML special characters...")
# Apply the function to the entire DataFrame
df = df.applymap(decode_html_special_chars)
print("Completed successfully!\n")

# Perform Token Blocking
print("Performing Token Blocking...")
block_index = token_blocking(df)
print("Completed successfully!\n")

# Create a list to store keys to print as sample
keys_to_print = []
# Iterate over each value (set of IDs) in the blocking index
for token_key, ID_value in block_index.items():
    # Each block should contain at least two entities, otherwise add the key to the keys_to_delete list
    if 10 <= len(ID_value) <= 100:
        keys_to_print.append(token_key)
    if len(keys_to_print) == 10:
        break
# Create a new dictionary containing only the desired keys and values
filtered_dict = {key: value for key, value in block_index.items() if key in keys_to_print}
# Display the sample index
print("Sample print of 10 mid-size blocks")
print("{'Blocking Key': {Entities}}")
print("----------------------------")
pprint(filtered_dict, width=120, compact=True)

# Write the resulting index to a file
print("\nWriting to a file, named `blocking_index_print.txt`, to be able to see the entire blocking index if needed...")
f = open('blocking_index_print.txt', 'w', encoding="utf-8")
with redirect_stdout(f):
    pprint(block_index, width=120, compact=True)
f.close()
print("Completed!")
print("\n--------------------------\n")


# TASK B ###############################################################################################################
# Define the function calculate_comparisons that takes a blocking index as input
def calculate_comparisons(blocking_index):
    # Initialize a variable to keep track of the total number of comparisons
    total_comparisons = 0
    # Iterate over each value (set of IDs) in the blocking index
    for value in blocking_index.values():
        # Calculate the number of unique entities in the block by getting the set's length
        num_entities = len(value)
        # Calculate the number of pairwise comparisons that can be made within the block
        # using the combination formula from the math module (math.comb)
        comparisons_in_block = comb(num_entities, 2)
        # Add the number of comparisons within this block to the total number of comparisons
        total_comparisons += comparisons_in_block
    # Return the total number of comparisons across all blocks in the blocking index
    return total_comparisons


# Calculate the number of comparisons for the given blocking index (block_index)
print("Calculating all pair-wise comparisons...")
num_comparisons = calculate_comparisons(block_index)
# Print the result, which represents the total number of pair-wise comparisons
print("Number of comparisons:", num_comparisons)
print("\n--------------------------\n")


# TASK C ###############################################################################################################
# Define a class named SQLiteGraph
class SQLiteGraph:

    # Constructor for the SQLiteGraph class
    def __init__(self, database_file):
        # Initialize a connection to an SQLite database using the provided file path
        # The database_file parameter represents the path to the SQLite database file
        # This connection will be used to interact with the SQLite database
        self.conn = sqlite3.connect(database_file)

    # Function to add nodes and edges to the SQLite database from a blocking index
    def add_nodes_and_edges_from_token_blocking_dict(self, blocking_index):
        # Establish a connection to the SQLite database
        conn = self.conn
        c = conn.cursor()
        # Drop the 'nodelist' and 'edgelist' tables if they exist
        c.execute('''DROP TABLE IF EXISTS nodelist''')
        c.execute('''DROP TABLE IF EXISTS edgelist''')
        # Create the 'nodelist' table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS nodelist (id INTEGER)''')
        # Create the 'edgelist' table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS edgelist
                     (source INTEGER, target INTEGER)''')
        # Begin a transaction
        c.execute('BEGIN TRANSACTION')
        # Progress bar to track the insertion progress
        pbar = tqdm(total=num_comparisons, desc='Progress')
        # Initialize an empty set to store unique nodes (entities)
        node_set = set()
        # Iterate over each value (list of IDs) in the blocking index
        for value in blocking_index.values():
            # Get the union (distinct IDs) of the sets node_set and the IDs contained in each block
            # and store it in the node_set
            node_set.union(value)
            # Use itertools.combinations to generate unique pairs within the current block
            block_pairs = set(combinations(value, 2))
            # Insert the pairs into the 'edgelist' table in the database
            c.executemany('INSERT INTO edgelist (source, target) VALUES (?, ?)', block_pairs)
            # Update the progress bar to reflect the number of inserted pairs
            pbar.update(len(block_pairs))
            # Clear the block_pairs set for memory efficiency
            block_pairs.clear()
        # Insert the unique nodes into the 'nodelist' table in the database
        c.executemany('INSERT INTO nodelist (id) VALUES (?)', node_set)
        # Commit the changes to the database
        conn.commit()
        # Set the progress bar description to 'Committing...' before closing it
        pbar.set_description('Committing...')
        # Close the progress bar
        pbar.set_description('Completed')
        pbar.close()

    # Function to run SELECT query on the SQLite database
    def query_graph(self, query):
        # Establish a connection to the SQLite database
        conn = self.conn
        c = conn.cursor()
        # Execute the SELECT query
        c.execute(query)
        # Fetch all rows from the query result
        rows = c.fetchall()
        # Return the query result
        return rows

    # Function to check if the number of edges in the graph (comparisons) is the same as a specified number
    def is_num_edges_equal_to(self, number_comparisons):
        # Define the SELECT query to count the number of rows (edges) in the 'edgelist' table
        query = 'SELECT COUNT(*) FROM edgelist'
        # Call the function to run the SELECT query on the graph database and retrieve the result
        result = self.query_graph(query)
        # Return a boolean indicating whether the number of edges in the database is equal to 'number_comparisons'
        return result[0][0] == number_comparisons

    # Function to apply Common Blocks Scheme (CBS) weights to the edges
    def apply_cbs_weighting_scheme(self):
        # Establish a connection to the SQLite database
        conn = self.conn
        c = conn.cursor()
        # Progress bar initialization and update
        pbar = tqdm(total=4, desc='Progress', mininterval=0.01)
        pbar.update(1)
        # Define the SELECT query to create a new table 'edgelist_weighted' with weighted edges
        c.execute('CREATE TABLE edgelist_weighted AS \
        SELECT source, target, COUNT(*) as weight FROM edgelist GROUP BY source, target')
        pbar.update(1)
        # Drop the existing 'edgelist' table to replace it with the weighted version
        c.execute('DROP TABLE edgelist')
        pbar.update(1)
        # Rename the 'edgelist_weighted' table to 'edgelist'
        c.execute('ALTER TABLE edgelist_weighted RENAME TO edgelist')
        pbar.update(1)
        # Commit the changes to the database
        pbar.set_description('Committing...')
        conn.commit()
        # Close the progress bar
        pbar.set_description('Completed')
        pbar.close()

    # Function to prune edges if their weight is lower than a specified limit
    def prune_edges_with_weight_lower_than(self, limit):
        # Establish a connection to the SQLite database
        conn = self.conn
        c = conn.cursor()
        # Define query to delete edges with weight lower than the specified 'limit'
        c.execute('DELETE FROM edgelist WHERE weight < ?', (limit,))
        # Commit changes to the database
        conn.commit()

    # Function to retrieve the graph's number of edges
    def number_of_edges(self):
        # Define SELECT query to count the number of rows in the 'edgelist' table
        query = 'SELECT COUNT(*) FROM edgelist'
        # Call the function to run the SELECT query on the database and retrieve the result
        result = self.query_graph(query)
        # Return the number of edges, which is extracted from the query result
        return result[0][0]

    # Function to retrieve the graph's number of nodes
    def number_of_nodes(self):
        # Define SELECT query to count the number of rows in the 'nodelist' table
        query = 'SELECT COUNT(*) FROM nodelist'
        # Call the function to run the SELECT query on the database and retrieve the result
        result = self.query_graph(query)
        # Return the number of nodes, which is extracted from the query result
        return result[0][0]

    # Method to close the connection established with the SQLite database
    def close_connection(self):
        # Close the connection to the SQLite database
        self.conn.close()


# Define the SQLite Graph Database file path
db_file = 'my_graph_database.db'
print("Creating the Meta-Blocking Graph...\n")
# Create the graph by initializing an instance of the SQLiteGraph class
graph = SQLiteGraph(db_file)

# Phase 1: Graph Building
print("1. Graph Building Phase: Adding all nodes (entities) and edges (comparisons) to the graph...")
# Add nodes and edges to the graph using the blocking index generated from token blocking
graph.add_nodes_and_edges_from_token_blocking_dict(block_index)

# Check if the number of graph edges is the same as the number of comparisons in Task B
print('\nCheck: Is the number of graph edges the same as the number of comparisons of Task B?:')
print(graph.is_num_edges_equal_to(num_comparisons))

# Phase 2: Edge Weighting
print("\n2. Edge Weighting Phase: Applying CBS Weighting Scheme...")
# Apply the CBS Weighting Scheme to the graph
print(graph.apply_cbs_weighting_scheme())

# Phase 3: Graph Pruning
# Set a threshold for pruning edges with weight below this value
threshold = 2
print("\n3. Graph Pruning Phase: Pruning edges with weight <", threshold, "...")
# Remove edges from the graph that have weight less than the specified threshold
graph.prune_edges_with_weight_lower_than(threshold)
print("Completed!")

# Check if the number of graph edges (after pruning) is the same as the initial number of comparisons (Task B)
print('Check: Is the number of graph edges (after pruning) the same as the initial number of comparisons (Task B)?:')
print(graph.is_num_edges_equal_to(num_comparisons))

# Phase 4: Block Collecting
print("\n4. Block Collecting Phase: Count every retained edge...")
# Count the number of edges after pruning to get the final number of comparisons
print('Final number of comparisons (after edge pruning):', graph.number_of_edges())

# Close the connection to the graph
print("\nClosing connection with the graph...\n")
graph.close_connection()
print("Closed!\n")
print("--------------------------\n")


# TASK D ###############################################################################################################
# Function that calculates the Jaccard similarity between two entities in a DataFrame based on their titles,
# given their IDs
def jaccard_similarity(dataframe, id1, id2):
    # Extract the titles of entities with ID1 and ID2 from the DataFrame
    title1 = dataframe.loc[dataframe['id'] == id1, 'title']
    title2 = dataframe.loc[dataframe['id'] == id2, 'title']
    # Convert the titles to lowercase and split them into sets of words
    set1 = set(str(title1.values[0]).lower().split())
    set2 = set(str(title2.values[0]).lower().split())
    # Calculate the intersection of the two sets (common words)
    intersection = len(set1.intersection(set2))
    # Calculate the union of the two sets (total unique words)
    union = len(set1.union(set2))
    # Calculate the Jaccard similarity by dividing the intersection by the union
    jaccard_sim = intersection / union
    # Return the Jaccard similarity value
    return jaccard_sim


# Test the function by calculating the Jaccard similarity of entities with ID 10 and 810
print("The Jaccard similarity of entities with ID 10 and 810 is:")
print(jaccard_similarity(df, 10, 810))
print("\n--------------------------")
print("END")
print("BYE-BYE")
