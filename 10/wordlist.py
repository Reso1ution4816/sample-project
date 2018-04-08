import textwrap
import pickle
import os


class WordEntry:
    """
    A single word entry in the wordbook
    """

    def __init__(self, word, meaning):
        self.word = word
        self.meaning = meaning
        self.num_correct = 0
        self.num_correct_in_row = 0
        self.is_last_time_correct = False

    def check_word(self, guess):
        if guess.strip().lower() == self.word.strip().lower():
            self.num_correct += 1
            if self.is_last_time_correct or self.num_correct == 1:
                self.num_correct_in_row += 1
            self.is_last_time_correct = True
            return 'correct'
        else:
            self.num_correct_in_row = 0
            self.is_last_time_correct = False
            return 'incorrect'

    def is_familiar(self):
        return self.num_correct >= 5 or self.num_correct_in_row >= 3

    def get_meaning(self):
        # every line contains only 70 characters
        return '\n'.join(textwrap.wrap(self.meaning, width=70))

    def get_counts(self):
        return 'correct counts: %s\nmax correct counts in a row: %s\nis last time correct:%s' % (
            self.num_correct, self.num_correct_in_row, self.is_last_time_correct
        )

    def __str__(self):
        """
        String representation of a WordEntry
        :return: String representation
        """
        return '%s\n\t%s\n\t%s\n' % (self.word, self.get_meaning(), self.get_counts())


class WordBook:
    def __init__(self):
        self.entries = set()

    def add_entry(self, *args):
        self.entries.add(WordEntry(*args))

    def list_unfamiliar(self):
        for we in self.entries:
            if not we.is_familiar():
                print(we)

    def word_test(self):
        for we in self.entries:
            if not we.is_familiar():
                print('\n' + '-' * 70)
                print(we.get_meaning())
                guess = input('-' * 70 + '\n what\'s the word?\n>>> ')
                print('that was', we.check_word(guess))
                print('-' * 70)


if __name__ == '__main__':
    save_file = 'wordbook_dump.pkl'

    if os.path.exists(save_file):
        wb = pickle.load(open(save_file, mode='rb'))
    else:
        wb = WordBook()
        # only for testing purpose
        wb.add_entry('variance',
                     'In probability theory and statistics, variance is the expectation of the squared deviation of a '
                     'random variable from its mean. Informally, it measures how far a set of (random) numbers are '
                     'spread out from their average value.')
        wb.add_entry('deviation',
                     'A deviation that is a difference between an observed value and the true value of a quantity of '
                     'interest (such as a population mean) is an error and a deviation that is the difference between '
                     'the observed value and an estimate of the true value (such an estimate may be a sample mean) '
                     'is a residual.')

    userinput = ''
    while userinput != 'offyougo':
        print('-' * 70)
        print('Welcome to MemoryLoss(TM) Vocabulary Builder!')
        print('Please entry your choice:')
        print('0. type: add      --- Add a new word, ')
        print('1. type: luf      --- List unfamiliar words')
        print('2. type: test     --- Test your memory on unfamiliar words, ')
        print('3. type: save     --- Save your vocabulary list')
        print('4. type: offyougo --- Exit and save')
        print('-' * 70)
        userinput = input('type your choice and press ENTER:')
        print('-' * 70)
        if userinput == 'add':
            word = input('type the word:')
            print('-' * 70)
            meaning = input('type the meaning:')
            wb.add_entry(word, meaning)
        elif userinput == 'luf':
            wb.list_unfamiliar()
            input('Type any key to continue')
        elif userinput == 'test':
            wb.word_test()
            input('Type any key to continue')
        elif userinput == 'save':
            pickle.dump(wb, open(save_file, mode='wb'))

    pickle.dump(wb, open(save_file, mode='wb'))
    print('\n' + '-' * 70)
    print('MemoryLoss(TM) Vocabulary Builder shut down gracefully.')
