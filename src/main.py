import tkinter as tk
from tkinter import messagebox
import subprocess


def run_script(script_path):
    try:
        subprocess.run([script_path], check=True, shell=True)
        messagebox.showinfo("Success", f"Script {script_path} executed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"Failed to execute {script_path}. Please check the script path or environment.")


root = tk.Tk()
root.title("Project Menu")
root.geometry("400x400")


menu_options = [
    ("Scraping", [
        ("Main Scrape", "C:/Users/user/PycharmProjects/GMR/src/1.Scrape_IMDB_7-10/1.Main_Scrape.py"),
        ("Additional Scrape", "C:/Users/user/PycharmProjects/GMR/src/1.Scrape_IMDB_7-10/2.Additional_Scrape.py"),
        ("Merge", "C:/Users/user/PycharmProjects/GMR/src/1.Scrape_IMDB_7-10/3.Merge.py")
    ]),
    ("Cleaning", [
        ("Clean", "C:/Users/user/PycharmProjects/GMR/src/2.Cleaning/4.Cleaning.py")
    ]),
    ("Normalize", [
        ("Normalize", "C:/Users/user/PycharmProjects/GMR/src/3.Normalize_Comparison/5.Normalize.py")
    ]),
    ("Clustering", [
        ("Standard Agglomerative / KMeans", "C:/Users/user/PycharmProjects/GMR/src/3.Normalize_Comparison/6.Comparison.py")
    ]),
    ("NN Clustering", [
        ("Encode NN clustering", "C:/Users/user/PycharmProjects/GMR/src/4.Neuro/7.Comparison_Encode.py")
    ]),
    ("Random Forest", [
        ("Random Forest", "C:/Users/user/PycharmProjects/GMR/src/5.Random_Forest/Random_forest.py")
    ]),
    ("Generate Random Movies", [
        ("Generate Movies", "C:/Users/user/PycharmProjects/GMR/src/6.Recommendation/3.Generate_movies.py"),
        ("Cleaning", "C:/Users/user/PycharmProjects/GMR/src/6.Recommendation/4.Cleaning.py"),
        ("Normalize", "C:/Users/user/PycharmProjects/GMR/src/6.Recommendation/5.Normalize.py"),
        ("Encode", "C:/Users/user/PycharmProjects/GMR/src/6.Recommendation/6.Encode.py")
    ]),
    ("Recommendation", [
        ("Cosine Similarity", "C:/Users/user/PycharmProjects/GMR/src/6.Recommendation/7.Compare/1.Cosing_Similarity.py")
    ])
]

def create_buttons(menu_frame, menu):
    for item_name, script_path in menu:
        button = tk.Button(menu_frame, text=item_name, width=30, command=lambda path=script_path: run_script(path))
        button.pack(pady=5)

def create_menu(menu_name, menu):
    frame = tk.LabelFrame(root, text=menu_name, padx=10, pady=10)
    frame.pack(padx=10, pady=10, fill="both", expand="true")
    create_buttons(frame, menu)

for menu_name, menu in menu_options:
    create_menu(menu_name, menu)

root.mainloop()
