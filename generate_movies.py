import csv
import random
import datetime

genres = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Fantasy", "Horror", "Mystery", "Crime"]
companies = ["Warner Bros", "Universal Pictures", "Paramount Pictures", "20th Century Fox", "Sony Pictures", "Walt Disney Pictures", "Lionsgate", "New Line Cinema", "Marvel Studios", "DreamWorks", "A24", "Pixar"]
countries = ["USA", "UK", "India", "Canada", "Australia", "France", "Germany", "South Korea", "Japan", "China"]
languages = ["en", "hi", "fr", "ko", "ja", "de", "es", "zh", "it", "ru"]

actors = [
    "Tom Hanks", "Leonardo DiCaprio", "Meryl Streep", "Brad Pitt", "Scarlett Johansson", 
    "Harrison Ford", "Denzel Washington", "Jennifer Lawrence", "Johnny Depp", "Emma Stone", 
    "Robert Downey Jr.", "Chris Evans", "Chris Hemsworth", "Mark Ruffalo", "Samuel L. Jackson", 
    "Will Smith", "Natalie Portman", "Anne Hathaway", "Hugh Jackman", "Christian Bale", 
    "Morgan Freeman", "Matt Damon", "Angelina Jolie", "Nicole Kidman", "Charlize Theron", 
    "Joaquin Phoenix", "Ryan Gosling", "Emma Watson", "Daniel Craig", "Tom Cruise", 
    "Keanu Reeves", "Chris Pratt", "Zendaya", "Tom Holland", "Margot Robbie", 
    "Florence Pugh", "Timothée Chalamet", "Anya Taylor-Joy", "Michael B. Jordan", "Chadwick Boseman", 
    "Oscar Isaac", "Pedro Pascal", "Viola Davis", "Ryan Reynolds", "Gal Gadot", 
    "Jason Momoa", "Henry Cavill", "Ben Affleck", "Amy Adams", "Jessica Chastain"
]

directors = [
    "Steven Spielberg", "Christopher Nolan", "Martin Scorsese", "Quentin Tarantino", "James Cameron", 
    "Peter Jackson", "Ridley Scott", "David Fincher", "Denis Villeneuve", "Greta Gerwig", 
    "Bong Joon Ho", "Alfonso Cuarón", "Wes Anderson", "Paul Thomas Anderson", "Guillermo del Toro", 
    "Taika Waititi", "Jordan Peele", "Edgar Wright", "Damien Chazelle", "Kathryn Bigelow", 
    "Sam Mendes", "George Miller", "Zack Snyder", "J.J. Abrams", "Patty Jenkins", 
    "James Gunn", "Rian Johnson", "Ryan Coogler", "Jon Favreau", "Matt Reeves"
]

adjectives = ["Dark", "Silent", "Hidden", "Last", "First", "Final", "Lost", "Golden", "Red", "Black", "White", "Blue", "Iron", "Steel", "Shattered", "Broken", "Fallen", "Rising", "Invisible", "Deadly", "Secret", "Forgotten", "Eternal", "Blind", "Cursed", "Midnight", "Quantum", "Shadow", "Neon", "Cyber"]
nouns = ["Knight", "Thunder", "Shadow", "Hero", "Legend", "Warrior", "Dawn", "City", "Mountain", "River", "Storm", "Sky", "Star", "Heart", "Soul", "Mission", "Journey", "Escape", "Dream", "Secret", "Echo", "Whisper", "Sword", "King", "Queen", "Empire", "Galaxy", "Protocol", "Syndicate", "Illusion"]
formats = [
    "The {adj} {noun}",
    "{adj} {noun}",
    "{noun} of the {noun}",
    "The {noun} {noun}",
    "{noun}: {adj} {noun}"
]

def generate_title():
    fmt = random.choice(formats)
    return fmt.format(
        adj=random.choice(adjectives),
        noun=random.choice(nouns)
    ).title()

rows = []
used_titles = set()
header = ["id", "title", "budget", "revenue", "genres", "runtime", "release_date", "production_companies", "production_countries", "cast", "director", "vote_average", "vote_count", "popularity", "original_language"]

for i in range(1, 251):
    while True:
        title = generate_title()
        if title not in used_titles:
            used_titles.add(title)
            break
            
    budget = random.randint(100_000, 300_000_000)
    
    # 1 to 3 genres
    num_genres = random.randint(1, 4)
    if num_genres > 3: num_genres = 3
    movie_genres = ", ".join(random.sample(genres, k=num_genres))
    
    runtime = random.randint(80, 180)
    
    start_date = datetime.date(1990, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    release_date = start_date + datetime.timedelta(days=random_number_of_days)
    release_str = release_date.strftime("%Y-%m-%d")
    
    # 1 to 2 production companies
    num_companies = random.choice([1, 1, 2, 2])
    movie_companies = ", ".join(random.sample(companies, k=num_companies))
    
    # 1 to 2 countries
    num_countries = random.choice([1, 1, 1, 1, 2])
    movie_countries = ", ".join(random.sample(countries, k=num_countries))
    
    # 3 to 5 cast members
    num_cast = random.randint(3, 5)
    movie_cast = ", ".join(random.sample(actors, k=num_cast))
    
    movie_director = random.choice(directors)
    
    vote_average = round(random.uniform(3.0, 9.5), 1)
    vote_count = random.randint(50, 10000)
    popularity = round(random.uniform(5.0, 100.0), 2)
    
    movie_lang = random.choices(languages, weights=[0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025], k=1)[0]
    
    # Calculate revenue logically
    # Revenue depends on budget, popularity, and vote_average
    # Popularity factor: higher popularity means higher multiplier
    popularity_factor = popularity / 40.0 # From 0.125 to 2.5
    # Vote factor: good votes increase revenue, bad votes decrease
    vote_factor = (vote_average - 5.0) / 4.0 # Range approx -0.5 to 1.125
    # Add some randomness
    noise = random.uniform(0.7, 1.3)
    
    base_multiplier = max(0.01, popularity_factor + vote_factor + noise) # min 0.01 to avoid negatives or zero
    
    # Sometimes big budget movies flop, sometimes small budgets make huge amounts (outliers)
    rand_chance = random.random()
    if rand_chance < 0.05: # 5% chance of breakout hit
        base_multiplier *= random.uniform(3.0, 10.0)
    elif rand_chance < 0.10: # 5% chance of massive flop
        base_multiplier *= random.uniform(0.1, 0.3)
        
    revenue = int(budget * base_multiplier)
    
    rows.append([
        i, title, budget, revenue, movie_genres, runtime, release_str,
        movie_companies, movie_countries, movie_cast, movie_director,
        vote_average, vote_count, popularity, movie_lang
    ])

with open("c:/Users/Admin/OneDrive/Desktop/ML LEARN/moviedata.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Generated moviedata.csv successfully with 250 rows.")
