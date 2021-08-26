import itertools
from concurrent.futures import ThreadPoolExecutor
from time import time

import pandas as pd
import psycopg2
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('tazi-se-interview-project-data.csv', index_col=0)


class PopulatePostgres:
    def __init__(self, df):
        self.df = df
        self.con = psycopg2.connect(host=5432, user='yusuf', password='yusuf123', dbname='tazi_assignment')

    def read_chunks(self):
        time.sleep(2)
        yield df.iloc[:100, :]

    def populate_sql(self):
        self.read_chunks().to_sql('tazi', self.con, if_exists='append')

    def read_from_database(self):
        # Waiting for the database to populate for the initial confusion matrix calculation
        time.sleep(20)

        self.df = pd.io.sql.read_sql('SELECT * FROM tazi', self.con)
        return df

class HasNextIterator:
    def __init__(self, it):
        self._it = iter(it)
        self._next = None

    def __iter__(self):
        return self

    def has_next(self):
        if self._next:
            return True
        try:
            self._next = next(self._it)
            return True
        except StopIteration:
            return False

    def next(self):
        if self._next:
            ret = self._next
            self._next = None
            return ret
        elif self.has_next():
            return self.next()
        else:
            raise StopIteration()


class StreamingData:
    
    def __init__(self, weights, chunk_size):
        self.weights = weights
        self.chunk_size = chunk_size

        self.con = psycopg2.connect(host=5432, user='yusuf', password='yusuf123', dbname='tazi_assignment')
        self.df = PopulatePostgres().read_from_database()
        self.iterator = HasNextIterator(itertools.count(0))

    @staticmethod
    def choose_a_or_b(row):
        if row['agg_A'] > row['agg_B']:
            row['choice'] = 'A'
        else:
            row['choice'] = 'B'
        return row

    def confusion_matrix(self):

        window_dict = {}
        confusion_matrix_dict = {}

        while self.iterator.has_next():
            a = self.iterator.next()

            window_dict[f'window_{a}'] = self.df.iloc[a-1:self.chunk_size+a-1, :]
            window = window_dict[f'window_{a}']

            window['agg_A'] = self.weights[0] * window['model1_A'] + self.weights[1] * window['model2_A'] + \
                              self.weights[2] * window['model3_A']
            window['agg_B'] = self.weights[0] * window['model1_B'] + self.weights[1] * window['model2_B'] + \
                              self.weights[2] * window['model3_B']
            window = window.apply(self.choose_a_or_b, axis=1)

            window_choice_a = window[window['choice'] == 'A']
            window_choice_b = window[window['choice'] == 'B']

            confusion_matrix_dict[f'confusion_matrix_{a}'] = pd.DataFrame()

            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['A', 'A'] = sum(
                window_choice_a['choice'] == window_choice_a['given_label'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['A', 'B'] = sum(
                window_choice_a['choice'] != window_choice_a['given_label'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['B', 'A'] = sum(
                window_choice_b['choice'] != window_choice_b['given_label'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['B', 'B'] = sum(
                window_choice_b['choice'] == window_choice_b['given_label'])

            confusion_matrix_dict[f'confusion_matrix_{a}'].to_sql(f'tazi_conf_matrix_{a}', self.con, if_exists='append')
        return confusion_matrix_dict


    def clear_matrix_data(self):
        while self.iterator.has_next():
            a = self.iterator.next()
            cur = self.con.cursor()
            cur.execute(f'DROP TABLE tazi_conf_matrix_{a};')


class SuperClass(PopulatePostgres, StreamingData):
    pass


def main1():
    sc = SuperClass()
    sc.read_chunks()
    sc.populate_sql()
    sc.read_from_database()


def main2():
    sc = SuperClass([0.3, 0.4, 0.3], 1000)
    sc.confusion_matrix()
    time.sleep(10)
    sc.clear_matrix_data()


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(main1())
        executor.submit(main2())

