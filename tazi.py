import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('tazi-se-interview-project-data.csv', index_col=0)


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
    
    def __init__(self, weights, chunk_size, df):
        self.weights = weights
        self.chunk_size = chunk_size
        self.df = df

        self.iterator = HasNextIterator(range(len(df)-self.chunk_size+1))

    @staticmethod
    def choose_a_or_b(row):
        row['choice'] = None
        if row['agg_A'] > row['agg_B']:
            row['choice'] = 'A'
        else:
            row['choice'] = 'B'
        return row

    def window_dfs(self):

        return

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

            window_dict[f'window_{a}'] = df.iloc[a-1:self.chunk_size+a-1, :]
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
        return confusion_matrix_dict


if __name__ == '__main__':
    sd = StreamingData([0.3, 0.4, 0.3], 1000, df)
    print(sd.confusion_matrix())
