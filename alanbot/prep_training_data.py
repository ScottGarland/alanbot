# We need to create files that are basically "parent" and "reply" text files, where each line is the sample.
# So line 15 in the parent file is the parent comment, and then line 15 in the reply file is the response to 
# line 15 in the parent file.

import sqlite3
import pandas as pd


timeframe = '2015-01'
connection = sqlite3.connect('{}.db'.format(timeframe))
curs = connection.cursor()
limit = 5000 # can adjust this for sample data
last_unix = 0 # buffer through database, find last unix timestamp, next pull unix must be greater than last unix etc
cursor_length = limit # shows when done
counter = 0 # helpful counter for debugging
test_done = False # we want an initial sample as a test


def open_content(filename):
    with open('{}.from'.format(filename),'a', encoding='utf8') as fp:
                for content in dataframe['parent'].values:
                    fp.write(content + '\n')

    with open('{}.to'.format(filename),'a', encoding='utf8') as fp:
                for content in dataframe['comment'].values:
                    fp.write(str(content) + '\n')


while cursor_length == limit:

        dataframe = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix, limit), connection)
        last_unix = dataframe.tail(1)['unix'].values[0] # setting the last_unix value appropriately
        cursor_length = len(dataframe) # this should be the length of the limit (5000)

        if not test_done:

            open_content('test')

            test_done = True # the test has finished

        else:
            
            open_content('train')
        
        counter += 1
        if counter % 20 == 0:
            print(counter*limit, 'rows completed') # this will print every 100000 rows completed
