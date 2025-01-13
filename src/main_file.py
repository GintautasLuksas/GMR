import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

df = pd.read_csv('imdb_movies.csv')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
df = df.where(pd.notnull(df), None)

def reload_data():
    subprocess.run(["python", "IMDb_collector.py"])
    global df
    df = pd.read_csv('imdb_movies.csv')  # Reload the CSV file
    df = df.where(pd.notnull(df), None)
    display_main_menu()

def show_top_10_movies_by_rating():
    top_10_movies = df.nlargest(10, 'Rating')
    plot_barh(top_10_movies['Title'], top_10_movies['Rating'], 'teal', 'Top 10 Movies by Rating', 'Rating', 'Movie Title')

def show_top_10_movies_by_rating_amount():
    df['Rating Amount'] = df['Rating Amount'].astype(float)
    df_sorted = df.sort_values('Rating Amount', ascending=False).head(10)
    plot_barh(df_sorted['Title'], df_sorted['Rating Amount'], 'purple', 'Top 10 Movies by Rating Amount', 'Rating Amount', 'Movie Title')

def show_top_10_movies_by_length():
    top_10_movies = df.nlargest(10, 'Duration (minutes)')
    plot_barh(top_10_movies['Title'], top_10_movies['Duration (minutes)'], 'gold', 'Top 10 Movies by Duration', 'Duration (minutes)', 'Movie Title')

def show_year_vs_rating():
    avg_rating_by_year = df.groupby('Year')['Rating'].mean().dropna()
    plot_line(avg_rating_by_year.index, avg_rating_by_year.values, 'blue', 'Average Rating by Year', 'Year', 'Average Rating')

def show_year_vs_length():
    avg_length_by_year = df.groupby('Year')['Duration (minutes)'].mean().dropna()
    plot_line(avg_length_by_year.index, avg_length_by_year.values, 'green', 'Average Length by Year', 'Year', 'Average Length (minutes)')

def show_year_vs_rating_amount():
    avg_rating_amount_by_year = df.groupby('Year')['Rating Amount'].mean().dropna()
    plot_line(avg_rating_amount_by_year.index, avg_rating_amount_by_year.values, 'purple', 'Average Rating Amount by Year', 'Year', 'Average Rating Amount')

def show_other_plots(plot_number):
    if plot_number == 1:
        plot_bar(df['Year'].value_counts().sort_index(), 'skyblue', 'Number of Movies by Year', 'Year', 'Number of Movies')
    elif plot_number == 2:
        plot_hist(df['Rating'], 20, 'green', 'Distribution of Ratings', 'Rating', 'Frequency')
    elif plot_number == 3:
        plot_scatter(df['Year'], df['Rating'], 'blue', 'Scatter Plot: Year vs Rating', 'Year', 'Rating')
    elif plot_number == 4:
        df['Length Category'] = pd.cut(df['Duration (minutes)'], bins=[0, 90, 120, 150, float('inf')],
                                       labels=['Short', 'Medium', 'Long', 'Very Long'])
        plot_violin('Length Category', 'Rating', 'Rating Distribution by Duration Category', 'Duration Category', 'Rating', 'viridis')
    elif plot_number == 5:
        avg_rating_by_year = df.groupby('Year')['Rating'].mean()
        plot_line(avg_rating_by_year.index, avg_rating_by_year.values, 'orange', 'Average Rating by Year', 'Year', 'Average Rating')
    elif plot_number == 6:
        plot_countplot('Group', 'Number of Movies by Group', 'Group', 'Number of Movies', 'Set3')

def plot_barh(x_data, y_data, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.barh(x_data, y_data, color=color)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

def plot_line(x_data, y_data, color, title, xlabel, ylabel):
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar(x_data, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    x_data.plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hist(x_data, bins, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    x_data.plot(kind='hist', bins=bins, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_scatter(x_data, y_data, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color=color, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_violin(x_data, y_data, title, xlabel, ylabel, palette):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x=x_data, y=y_data, data=df, hue=x_data, palette=palette, inner='quartile', legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_countplot(x_data, title, xlabel, ylabel, palette):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=x_data, data=df, hue=x_data, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def display_main_menu():
    for widget in root.winfo_children():
        widget.destroy()

    label = tk.Label(root, text="Select a category to display:")
    label.pack()

    top10_button = tk.Button(root, text="Category: Top 10", command=display_top_10_submenu)
    top10_button.pack(pady=10)

    year_comparison_button = tk.Button(root, text="Category: Year Comparison", command=display_year_vs_submenu)
    year_comparison_button.pack(pady=10)

    other_plots_button = tk.Button(root, text="Others (All other plots)", command=display_other_plots_submenu)
    other_plots_button.pack(pady=10)

    reload_button = tk.Button(root, text="Reload Data", command=reload_data)
    reload_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=root.destroy)
    exit_button.pack(pady=10)


def display_top_10_submenu():
    for widget in root.winfo_children():
        widget.destroy()

    label = tk.Label(root, text="Select a Top 10 Movies option:")
    label.pack()

    button1 = tk.Button(root, text="Top 10 Movies by Rating", command=show_top_10_movies_by_rating)
    button1.pack(pady=10)

    button2 = tk.Button(root, text="Top 10 Movies by Rating Amount", command=show_top_10_movies_by_rating_amount)
    button2.pack(pady=10)

    button3 = tk.Button(root, text="Top 10 Movies by Length", command=show_top_10_movies_by_length)
    button3.pack(pady=10)

    back_button = tk.Button(root, text="Back to Main Menu", command=display_main_menu)
    back_button.pack(pady=10)

def display_year_vs_submenu():
    for widget in root.winfo_children():
        widget.destroy()

    label = tk.Label(root, text="Select a Year vs ... option:")
    label.pack()

    button1 = tk.Button(root, text="Average Rating by Year", command=show_year_vs_rating)
    button1.pack(pady=10)

    button2 = tk.Button(root, text="Average Length (in minutes) by Year", command=show_year_vs_length)
    button2.pack(pady=10)

    button3 = tk.Button(root, text="Average Rating Amount by Year", command=show_year_vs_rating_amount)
    button3.pack(pady=10)

    back_button = tk.Button(root, text="Back to Main Menu", command=display_main_menu)
    back_button.pack(pady=10)

def display_other_plots_submenu():
    for widget in root.winfo_children():
        widget.destroy()

    label = tk.Label(root, text="Select an Other Plot option:")
    label.pack()

    button1 = tk.Button(root, text="Number of Movies by Year", command=lambda: show_other_plots(1))
    button1.pack(pady=10)

    button2 = tk.Button(root, text="Distribution of Ratings", command=lambda: show_other_plots(2))
    button2.pack(pady=10)

    button3 = tk.Button(root, text="Rating Distribution by Duration Category", command=lambda: show_other_plots(4))
    button3.pack(pady=10)

    button4 = tk.Button(root, text="Average Rating by Year", command=lambda: show_other_plots(5))
    button4.pack(pady=10)

    button5 = tk.Button(root, text="Number of Movies by MovieGroup", command=lambda: show_other_plots(6))
    button5.pack(pady=10)

    back_button = tk.Button(root, text="Back to Main Menu", command=display_main_menu)
    back_button.pack(pady=10)


root = tk.Tk()
root.title('IMDB Movie Analysis')

display_main_menu()

root.mainloop()
