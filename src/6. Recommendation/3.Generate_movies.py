import random
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.IMDB710_4000.Main_Scrape import save_to_csv

# Random lists with 100 examples for text values
movie_titles = [
    'The Lost Time', 'Journey Beyond', 'Shadows of Tomorrow', 'Echoes in the Void', 'Whispers of Eternity',
    'The Fallen Kingdom', 'Frozen Hearts', 'Edge of Infinity', 'Into the Abyss', 'The Last Stand',
    'Dreams of the Silent', 'City of Illusions', 'Awaken the Storm', 'Lost in the Dark', 'Fires of the Past',
    'Echoes of the Fallen', 'Under the Blue Sky', 'Rising from Ashes', 'Veil of Darkness', 'Silent Requiem',
    'Wings of Destiny', 'Shattered Horizons', 'In the Realm of Gods', 'Unseen Forces', 'Lurking in the Shadows',
    'Time for Tomorrow', 'The Eternal Flame', 'Darkened Light', 'Beyond the Clouds', 'Whispers of Hope',
    'Rogue Waves', 'Frozen in Time', 'Moonlit Nights', 'Tales of the Forgotten', 'Beneath the Surface',
    'Broken Chains', 'Endless Nights', 'The Final Hour', 'City of Lost Souls', 'The Hidden Path',
    'Fading Memories', 'Threads of Fate', 'Awakening the Beast', 'Redemption Road', 'Voices of the Lost',
    'Mystery of the Ancient', 'Streets of the Underdog', 'Journey to the Unknown', 'Lost Stars', 'The Twilight Zone',
    'Dust of Time', 'Into the Wild', 'Forgotten Dreams', 'Burning Bridges', 'Whispers of the Wind',
    'Tides of Time', 'Forgotten Realm', 'Waking the Giant', 'Chasing the Dawn', 'Secrets of the Past',
    'Waking the Fallen', 'The Last Quest', 'A New Dawn', 'Stormfront', 'The Silent City', 'Embers of the Fallen',
    'Hearts in Battle', 'The Golden Empire', 'Beyond the Storm', 'Through the Flames', 'Unveiling the Unknown',
    'Out of the Dark', 'The Lost Chronicles', 'Crimson Legacy', 'The Warrior\'s Path', 'Rising Phoenix',
    'Voices from the Deep', 'Storm’s End', 'Dawn of the Brave', 'Into the Light', 'The Forgotten King',
    'Chasing Destiny', 'Soul of the Sea', 'The Wild Hunt', 'Battle of the Lost', 'Across the Void',
    'Reign of Chaos', 'A Tale of Shadows', 'The Fallen Star', 'Legacy of the Brave', 'Tears of the Earth',
    'Shattered Heart', 'The Silver Blade', 'Rise of the Hero', 'The Silent Hero', 'Unbroken Spirit',
    'Storm of Souls', 'Hidden Legends', 'Rise from the Ashes', 'The Last Warrior', 'Through the Shadows',
    'The Ghost Ship', 'End of the Line', 'The Final Hope', 'Echo of the Lost', 'Waves of Destiny',
    'Beneath the Stars', 'Path of Redemption', 'Dark Horizons', 'The Unseen Land', 'Crimson Sky'
]

groups = ['PG', 'PG-13', 'R', 'G']

directors = [
    'James Phoenix', 'Lily Moon', 'Hector Waters', 'Sarah Blaze', 'Jack Harper', 'Oliver Green', 'Eva Storm',
    'Luke Sutherland', 'Maya Cross', 'Adrian Wolf', 'Scarlett Reed', 'Finn Donovan', 'Tessa Bright', 'Zane Black',
    'Ella Rivers', 'Cole Weston', 'Olivia Grant', 'Leo Bellamy', 'Grace Bell', 'Daniel Turner', 'Rachel Knight',
    'Hunter Frost', 'Jade Crawford', 'Violet Winter', 'Max Steele', 'Chloe Fields', 'Nathanial Rivers', 'Isabella Stone',
    'Leo Hart', 'Camilla Steele', 'Victor Price', 'Ivy Monroe', 'Logan Flynn', 'Emily Quinn', 'Michael Cross',
    'Riley Lane', 'Sophia Gray', 'Ryan Cole', 'Cassandra Stone', 'David Ashford', 'Amelia Rivers', 'Caleb Black',
    'Eliot Hayes', 'Ruby Evans', 'Leo Cooper', 'Zoe Thompson', 'Julian Fox', 'Ava Morrow', 'Luke Rivers', 'Hannah Brooks',
    'Brandon Lane', 'Julian White', 'Eliza Drake', 'Zane Pierce', 'Sophie Blake', 'Oliver Gray', 'Sienna Bell',
    'Lily Walters', 'Chris Weston', 'Alexa Nash', 'Mason Knight', 'Jessica Burns', 'Wyatt Clarke', 'Ella Mendez',
    'Carter Holt', 'Anastasia Blake', 'Gabriel Bennett', 'Madeline Winter', 'Isaac Harper', 'Aiden Ross',
    'Vivian Fox', 'Ethan Drake', 'Olivia Stone', 'Evan Turner', 'Savannah King', 'Daniel Steele', 'Amos Bright',
    'Maya Walker', 'Joshua Hunt', 'Sasha Montgomery', 'Luke Pierce', 'Quinn Monroe', 'Abigail Stone', 'Jasper Hart',
    'Naomi Shaw', 'Tyler Matthews', 'Ryan Moon', 'Riley Rivers', 'Catherine Hale', 'Brayden Ford', 'Clara Wells',
    'Theodore West', 'Carla James', 'Kaitlyn Knox', 'Jackson Reyes', 'Xander Price', 'Leila Fisher', 'Samuel Burns'
]

stars = [
    'Chris Hemsworth', 'Scarlett Johansson', 'Tom Hardy', 'Jennifer Lawrence', 'Ryan Reynolds', 'Emma Stone',
    'Leonardo DiCaprio', 'Brad Pitt', 'Natalie Portman', 'Will Smith', 'Charlize Theron', 'Jake Gyllenhaal',
    'Margot Robbie', 'Chris Pratt', 'Ryan Gosling', 'Anne Hathaway', 'Tom Hanks', 'Meryl Streep', 'Dwayne Johnson',
    'Hugh Jackman', 'Gal Gadot', 'Matthew McConaughey', 'Johnny Depp', 'Sandra Bullock', 'Zoe Saldana', 'Idris Elba',
    'Emily Blunt', 'Bradley Cooper', 'Cate Blanchett', 'Keanu Reeves', 'Oscar Isaac', 'Charlize Theron', 'Ben Affleck',
    'Sophie Turner', 'James McAvoy', 'Harrison Ford', 'Benedict Cumberbatch', 'Rami Malek', 'Mark Ruffalo',
    'David Oyelowo', 'Jessica Chastain', 'Chiwetel Ejiofor', 'Michelle Williams', 'Mark Wahlberg', 'Tom Cruise',
    'Emma Watson', 'Chris Evans', 'Henry Cavill', 'Samuel L. Jackson', 'Margot Robbie', 'Jake Johnson', 'Tessa Thompson',
    'Viggo Mortensen', 'Olivia Wilde', 'Will Ferrell', 'Jack Black', 'Keira Knightley', 'Jessica Alba', 'Matthew Fox',
    'Kristen Stewart', 'Harrison Ford', 'Eddie Redmayne', 'Jared Leto', 'Samuel L. Jackson', 'Matt Damon',
    'Michael B. Jordan', 'Jake Gyllenhaal', 'Helen Mirren', 'Jason Momoa', 'Tom Hiddleston', 'Zac Efron', 'Amy Adams',
    'John Boyega', 'Michelle Pfeiffer', 'Bryan Cranston', 'Steve Carell', 'Naomi Watts', 'Mark Ruffalo', 'Paul Rudd',
    'Alicia Vikander', 'Channing Tatum', 'Zoe Kravitz', 'Ethan Hawke', 'Emma Roberts', 'Tommy Lee Jones', 'Jeff Bridges',
    'Martin Freeman', 'Lupita Nyong’o', 'Nicole Kidman', 'Michael Fassbender', 'Chris Pine', 'Tom Welling', 'Kate Winslet'
]

# Random short descriptions
short_descriptions = [
    'A young hero embarks on a journey to save the world.',
    'A story of love, loss, and redemption set against the backdrop of war.',
    'The search for a missing person takes an unexpected turn.',
    'A group of strangers discover they are linked by a shared secret.',
    'An ancient artifact holds the key to a forgotten civilization.',
    'In a dystopian future, the last hope for humanity lies in the hands of one.',
    'A detective uncovers a conspiracy that could change history forever.',
    'A scientist’s obsession leads to unforeseen consequences.',
    'Two unlikely allies must work together to prevent an apocalypse.',
    'A new technology threatens to change the fabric of society.',
    'A family must survive after a catastrophic event changes everything.',
    'A journey of self-discovery leads to unexpected adventures.',
    'A mysterious figure brings chaos to a quiet town.',
    'A young girl’s courage brings hope to those around her.',
    'A legendary hero must face their greatest challenge yet.',
    'An ancient evil is awakening, and it’s up to one person to stop it.',
    'A love story that transcends time and space.',
    'A war hero is forced to reckon with the consequences of their actions.',
    'A seemingly perfect world begins to unravel as secrets are revealed.',
    'A supernatural force threatens to destroy everything in its path.',
    'A man’s obsession with the past leads to his downfall.',
    'A group of rebels fight against an oppressive regime.',
    'A young woman must choose between loyalty and survival.',
    'A man’s life is turned upside down by a strange and mysterious event.',
    'A city on the brink of collapse is saved by an unlikely hero.',
    'A detective must solve a series of murders that seem to be connected.',
    'A scientist’s discovery could change the world forever.',
    'A historical figure is brought to life in a new and thrilling way.',
    'A woman’s search for justice leads her down a dangerous path.',
    'A team of explorers uncovers the secrets of a lost civilization.',
    'A politician’s dark past threatens to derail their career.',
    'A group of strangers find themselves trapped in a haunted mansion.',
    'An alien invasion forces humanity to band together for survival.',
    'A young couple’s honeymoon becomes a fight for survival.',
    'A mysterious benefactor offers a chance for redemption.',
    'A soldier must confront the horrors of war in a foreign land.',
    'A thief must outwit a group of dangerous criminals.',
    'A young man discovers he has extraordinary powers.',
    'A tragic accident changes the course of a family’s life forever.',
    'A young woman uncovers a hidden truth about her family’s past.',
    'A group of teenagers must confront their worst fears in order to survive.',
    'A scientist’s experiment goes terribly wrong, leading to disastrous results.',
    'A young boy’s dream leads him to a journey of self-realization.',
    'A band of misfits must save the world from an impending catastrophe.',
    'A man’s life is turned upside down by a mysterious phone call.',
    'A family must fight to survive in a world overrun by zombies.',
    'A once-proud king must reclaim his throne from a treacherous usurper.',
    'A group of adventurers search for a treasure hidden for centuries.',
    'A young man’s quest for vengeance leads to a dangerous confrontation.',
    'A soldier returns home only to find that everything has changed.',
    'A powerful corporation controls the fate of the world.',
    'A young woman discovers her true purpose in life.',
    'A group of friends discover that their vacation is not what it seems.',
    'A man’s struggle for survival takes him to the edge of insanity.',
    'A mysterious benefactor offers a fortune in exchange for a dangerous mission.',
    'A group of survivors must work together to outwit a deadly enemy.',
    'A woman’s past catches up to her in the form of a vengeful ex-lover.',
    'A brilliant scientist must race against time to prevent disaster.',
    'A secret society holds the key to an ancient mystery.',
    'A group of rebels fight for their freedom in a brutal war.',
    'A young woman is forced to confront her darkest fears.',
    'A detective uncovers a twisted conspiracy that threatens to destroy everything.',
    'A young boy’s innocent curiosity leads to an unexpected adventure.',
    'A ruthless criminal mastermind plots to take control of a city.',
    'A group of strangers must survive a deadly game of cat and mouse.',
    'A man’s search for his missing wife takes a dark and twisted turn.',
    'A couple’s perfect life is shattered by a shocking revelation.',
    'A legendary monster is awakened from centuries of slumber.',
    'A series of unexplainable events leads a man to question everything he knows.',
    'A group of strangers must trust each other to survive a deadly journey.',
    'A young woman embarks on a quest to find her missing father.',
    'A soldier’s loyalty is tested in a war torn country.',
    'A family is torn apart by a tragic accident.',
    'A woman’s journey to uncover the truth behind a family mystery.',
    'A young couple’s love story is threatened by the secrets they hide.',
    'A man’s obsession with an ancient artifact leads to disaster.',
    'A group of survivors must face a deadly virus that turns people into monsters.',
    'A young girl must fight to survive in a post-apocalyptic world.',
    'A scientist’s discovery could change the course of human history.',
    'A man must fight to protect his family from a deadly force.',
    'A powerful corporation threatens to destroy the environment.',
    'A detective uncovers a shocking truth about a series of murders.',
    'A group of rebels fight to overthrow a corrupt government.',
    'A woman’s search for her missing daughter leads to a dangerous conspiracy.',
    'A young boy’s life is turned upside down by a supernatural event.',
    'A scientist’s experiment goes terribly wrong, threatening humanity.',
    'A team of explorers discovers the secret to immortality.',
    'A woman’s quest for vengeance leads her to an unexpected discovery.',
    'A group of survivors must face their worst fears in order to escape.',
    'A woman’s life is changed forever by a chance encounter.',
    'A brilliant detective uncovers a dark secret that threatens to tear apart a family.',
    'A hero’s journey to save the world from destruction.',
    'A man’s pursuit of justice leads him down a dangerous path.',
    'A woman must confront her past in order to move forward.',
    'A soldier’s bravery saves his comrades in a battle for survival.',
    'A young girl’s quest to find her missing brother.',
    'A woman’s pursuit of happiness leads to unexpected twists.',
    'A man’s search for revenge takes him to the darkest corners of the earth.',
    'A lost treasure is the key to unlocking an ancient secret.'
]

# Create a DataFrame
df = pd.DataFrame({
    'Title': random.choices(movie_titles, k=100),
    'Year': [random.randint(1960, 2024) for _ in range(100)],
    'Rating': [round(random.uniform(7.2, 9.3), 1) for _ in range(100)],
    'Length (minutes)': [random.randint(80, 150) for _ in range(100)],  # Ensures length is between 80 and 150
    'Rating Amount': [random.randint(10000, 1000000) for _ in range(100)],  # Ensures rating amount is between 10,000 and 1,000,000
    'Metascore': [round(random.uniform(50.0, 100.0), 1) for _ in range(100)],  # Ensures Metascore is between 50.0 and 100.0
    'Group': random.choices(groups, k=100),
    'Short Description': random.choices(short_descriptions, k=100),
    'Directors': random.choices(directors, k=100),
    'Stars': random.choices(stars, k=100)
})

# Show the first 5 rows of the dataframe
print(df.head())

df.to_csv('movie_data.csv', index=False)
