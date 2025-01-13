# IMDb Movie Analysis

This application provides a graphical user interface for analyzing IMDb movie data using various plots. The data is initially read from a CSV file, and the application allows users to reload the data by running an external script.

## Features
- **Top 10 Movies by Rating**: Displays a bar plot of the top 10 movies by rating.
- **Top 10 Movies by Rating Amount**: Displays a bar plot of the top 10 movies by rating amount.
- **Top 10 Movies by Duration**: Displays a bar plot of the top 10 movies by duration.
- **Year vs Rating**: Displays a line plot showing the average rating by year.
- **Year vs Length**: Displays a line plot showing the average movie length by year.
- **Year vs Rating Amount**: Displays a line plot showing the average rating amount by year.
- **Other Plots**: Displays various other plots, including:
  - Number of Movies by Year
  - Distribution of Ratings
  - Scatter Plot: Year vs Rating
  - Rating Distribution by Duration Category
  - Number of Movies by Group

## File Structure
- **main.py**: The main script that runs the GUI application.
- **IMDb_collector.py**: The script to update the imdb_movies.csv file.
- **imdb_movies.csv**: The CSV file containing IMDb movie data.

## Dependencies
- **tkinter**: For creating the GUI and making it more fun.
- **pandas**: For data manipulation.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For making statistical graphics.
