import sqlite3
import json
from datetime import datetime


TRANSACTION_SIZE = 1000
DATA_LENGTH = 1000
BIN_SIZE = 50

timeframe = '2015-01'
sql_transaction = [] # instead of inserting rows one by one, we can use this for a large transaction
connection = sqlite3.connect('{}.db'.format(timeframe)) # database with 2015-01 as the title
curs = connection.cursor()


def create_table():
    curs.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

def format_data(data):
    """
    This function is used to format the data for a cleaner output that can be tokenized more easily
    """
    
    data = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return data

def data_filter(data):
    """
    This function applies a filter to the input data to makes sure it fits the criteria of being inserted into the database.
    """

    if len(data.split(' ')) > BIN_SIZE or len(data) < 1: # bin size can be adjusted here
        return False

    elif len(data) > DATA_LENGTH: # length size cut-off can be adjusted
        return False

    elif data == '[deleted]':
        return False

    elif data == '[removed]':
        return False
    
    else:
        return True # passes the filter and the data is acceptable

def find_parent_or_score(parent_id, score=False):
    """
    This function is to find the first parent of any comment.
    If comment_id in database matches another comment's parent_id, then match new comment w/ that parent_id.
    
    The second part of this function finds if parent_id already has a comment with a high score.
    These two are put into one function due to similar logic.

    score boolean value decides which sql string to use
    """

    try:
        if score == False :
            sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(parent_id) # find the parent
        if score == True:
            sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(parent_id) # find the existing high score comment

        curs.execute(sql)
        result = curs.fetchone()

        if result != None:
            return result[0]
        else:
            return False

    except Exception as e:
        print('parent_id', str(e))
        return False

def sql_insert_update(comment_id, parent_id, parent_data, body, subreddit, created_utc, score, update=False, parent_check=False):
    """
    Function made for inserting data into database via SQL queries.

    update is a boolean value turning on functionality for replacing a comment if one exists with a better score.
    parent is a boolean value for if there is a parent
    """

    if update == True:
        try:
            sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parent_id, comment_id, parent_data, body, subreddit, int(created_utc), score)
            transaction_builder(sql)
        except Exception as e:
            print('SQL UPDATE',str(e))
    
    else:
        try:
            if parent_check == True:
                sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parent_id, comment_id, parent_data, body, subreddit, int(created_utc), score)
                transaction_builder(sql)

            elif parent_check == False:
                sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parent_id, comment_id, body, subreddit, int(created_utc), score)
                transaction_builder(sql)
        except Exception as e:
            print('SQL INSERT PARENT',str(e))

def transaction_builder(sql):
    """
    Build the sql transaction until it's a certain size to keep consistent query execution time.
    """

    global sql_transaction # this global variable is to clear out the list at the start of the file
    sql_transaction.append(sql) # keep appending
    
    if len(sql_transaction) > TRANSACTION_SIZE:
        curs.execute('BEGIN TRANSACTION')
        for query in sql_transaction:
            try:
                curs.execute(query)
            except:
                pass

        connection.commit()
        sql_transaction = [] # now we want to clear the transaction queue


if __name__ == '__main__':
    create_table()
    row_counter = 0 # each row
    paired_rows = 0 # parent-child pair

    with open('raw_data/RC_2015-01', buffering=1000) as fp:
        for row in fp:
            #print(row) # you can use this line to make sure the data is being accessed correctly and outputting to

            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            comment_id = row['name']
            subreddit = row['subreddit']

            parent_data = find_parent_or_score(parent_id, score=False)

            if score >= 2: # at least one person saw the comment and upvoted it

                # for the current comment, we're going to see if there already exists comment replying to parent_id that has a score greater than current comment
                existing_score = find_parent_or_score(parent_id, score=True)
                if existing_score:
                    if score > existing_score: # insert the data in the db under this score threshold condition
                        if data_filter(body):
                            sql_insert_update(comment_id, parent_id, parent_data, body, subreddit, created_utc, score, update=True, parent_check=False)
                else:
                    if data_filter(body):
                        if parent_data:
                            sql_insert_update(comment_id, parent_id, parent_data, body, subreddit, created_utc, score, update=False, parent_check=True)
                            paired_rows += 1
                        
                        else:
                            sql_insert_update(comment_id, parent_id, parent_data, body, subreddit, created_utc, score, update=False, parent_check=False)
            
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))
