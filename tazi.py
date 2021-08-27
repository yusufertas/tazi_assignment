import itertools
from concurrent.futures import ThreadPoolExecutor
import time
from sqlalchemy import create_engine

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

filename = 'tazi-se-interview-project-data.csv'
engine = create_engine('postgresql://yusuf:yusuf123@localhost:5432/tazi_assignment')
engine.execute('CREATE TABLE IF NOT EXISTS tazi(id SERIAL PRIMARY KEY, given_label VARCHAR(1), "model1_A" NUMERIC, "model1_B" NUMERIC, "model2_A" NUMERIC, "model2_B" NUMERIC, "model3_A" NUMERIC, "model3_B" NUMERIC);')


class PopulatePostgres:

    def __init__(self, **kwargs):
        self.filename = filename
        super().__init__(**kwargs)

    def read_chunks(self):
        reader = pd.read_csv(self.filename, chunksize=100, index_col=0)
        for chunk in reader:
            time.sleep(1)
            chunk.to_sql('tazi', engine, if_exists='append')

    def read_from_database(self):
        # Waiting for the database to populate for the initial confusion matrix calculation
        time.sleep(10)
        dataframe = pd.io.sql.read_sql('SELECT * FROM tazi', engine)
        return dataframe


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

    def __init__(self, weights, chunk_size, **kwargs):
        self.weights = weights
        self.chunk_size = chunk_size

        self.streaming_data = PopulatePostgres().read_from_database()
        self.iterator = HasNextIterator(itertools.count(0))
        super().__init__(**kwargs)

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
            window_dict[f'window_{a}'] = self.streaming_data.iloc[a-1:self.chunk_size+a-1, :]
            window = window_dict[f'window_{a}']

            window['agg_A'] = self.weights[0] * window['model1_A'] + self.weights[1] * window['model2_A'] + \
                              self.weights[2] * window['model3_A']
            window['agg_B'] = self.weights[0] * window['model1_B'] + self.weights[1] * window['model2_B'] + \
                              self.weights[2] * window['model3_B']
            window = window.apply(self.choose_a_or_b, axis=1)

            window_choice_a = window[window['choice'] == 'A']
            window_choice_b = window[window['choice'] == 'B']

            confusion_matrix_dict[f'confusion_matrix_{a}'] = pd.DataFrame()
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['A', 'A'] = len(window_choice_a[window_choice_a['given_label']=='A'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['A', 'B'] = len(window_choice_a[window_choice_a['given_label']=='B'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['B', 'A'] = len(window_choice_b[window_choice_b['given_label']=='A'])
            confusion_matrix_dict[f'confusion_matrix_{a}'].loc['B', 'B'] = len(window_choice_b[window_choice_b['given_label']=='B'])

            confusion_matrix_dict[f'confusion_matrix_{a}'].to_sql(f'tazi_conf_matrix_{a}', engine, if_exists='replace')
            # print(confusion_matrix_dict[f'confusion_matrix_{a}'])

        return confusion_matrix_dict


    def clear_matrix_data(self):
        iterator = iter(itertools.count(1))
        loop_finished = False
        while not loop_finished:
            try:
                item = next(iterator)
                time.sleep(5)
                engine.execute(f'DROP TABLE tazi_conf_matrix_{item};')
            except StopIteration:
                loop_finished = True


class SuperClass(PopulatePostgres, StreamingData):
    def __init__(self, weights, chunk_size):
        super().__init__(weights=weights, chunk_size=chunk_size)


sc = SuperClass([0.1, 0.8, 0.1], 1000)


def main1():
    sc.read_chunks()
    sc.read_from_database()


def main2():
    sc.confusion_matrix()


def main3():
    sc.clear_matrix_data()


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(main1)
        executor.submit(main2)
        executor.submit(main3)
