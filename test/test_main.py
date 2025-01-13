import sys
import unittest
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

sys.path.append('../src')

import main_file as main

class TestIMDBAnalysisApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This method runs once before all tests, set up any initial configurations or data here
        cls.df = pd.DataFrame({
            'Title': ['Movie A', 'Movie B', 'Movie C'],
            'Rating': [8.5, 9.0, 7.5],
            'Rating Amount': [1000, 1500, 800],
            'Duration (minutes)': [120, 110, 130],
            'Year': [2010, 2015, 2020],
            'Group': ['Group1', 'Group2', 'Group1']
        })

    def test_show_top_10_movies_by_rating(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_barh') as mock_plot_barh:
            main.show_top_10_movies_by_rating()
            mock_plot_barh.assert_called_once()

    def test_show_top_10_movies_by_rating_amount(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_barh') as mock_plot_barh:
            main.show_top_10_movies_by_rating_amount()
            mock_plot_barh.assert_called_once()

    def test_show_top_10_movies_by_length(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_barh') as mock_plot_barh:
            main.show_top_10_movies_by_length()
            mock_plot_barh.assert_called_once()

    def test_show_year_vs_rating(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_line') as mock_plot_line:
            main.show_year_vs_rating()
            mock_plot_line.assert_called_once()

    def test_show_year_vs_length(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_line') as mock_plot_line:
            main.show_year_vs_length()
            mock_plot_line.assert_called_once()

    def test_show_year_vs_rating_amount(self):
        main.df = self.df  # Mocking the dataframe
        with patch('main_file.plot_line') as mock_plot_line:
            main.show_year_vs_rating_amount()
            mock_plot_line.assert_called_once()

    def test_display_main_menu(self):
        root = tk.Tk()
        with patch.object(root, 'mainloop'):
            with patch('main_file.display_main_menu'):
                main.display_main_menu()
                self.assertEqual(len(root.winfo_children()), 5)  # Check if all buttons and labels are created

    def test_reload_data(self):
        with patch('subprocess.run'):
            with patch('pandas.read_csv', return_value=self.df):
                main.reload_data()
                self.assertTrue(isinstance(main.df, pd.DataFrame))

    # Add more tests as needed for other functions...

if __name__ == '__main__':
    unittest.main()

