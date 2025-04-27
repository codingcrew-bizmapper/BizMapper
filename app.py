from flask import Flask, jsonify, send_from_directory, request, render_template, session, url_for, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from werkzeug.utils import secure_filename
import pandas as pd
from flask_cors import CORS
from flask_pymongo import PyMongo
import os
import re
import ast
from urllib.parse import unquote
from urllib.parse import quote
import openai
from datetime import timedelta


app = Flask(__name__)

# Enable CORS
CORS(app)
app.secret_key = os.urandom(24) 
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB URI setup
# app.config["MONGO_URI"] = "place the original mongo uri here"
mongo = PyMongo(app)

# Register a custom Jinja filter for URL encoding
@app.template_filter('url_encode')
def url_encode_filter(text):
    return quote(text)


# Serve static files
@app.route("/")
def home():
    user_logged_in = session.get('user_logged_in', False)
    return render_template('home.html', user_logged_in=user_logged_in)

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/profile")
def profile():
    if 'user_logged_in' in session:
        username = session['username']  # Get the username from the session
        
        # Fetch the user data from the database
        user = mongo.db.users.find_one({"username": username})

        if user:
            # Ensure favorites exist in user data
            favorites = user.get('favorites', [])
            
            # Render the profile page with user data
            return render_template('profile.html', user=user, favorites=favorites)
        else:
            # If the user is not found in the database, log them out
            session.clear()  # Clear the session
            return redirect(url_for('login'))
    else:
        # If the user is not logged in, redirect to login page
        return redirect(url_for('login'))

# Route for login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = mongo.db.users.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session['username'] = username  # Store username in session
            session['user_logged_in'] = True  # Maintain login status
            session['favorites'] = user.get('favorites', []) 
            return render_template("home.html", user_logged_in=True)
        else:
            return jsonify({"message": "Invalid credentials", "status": "error"}), 401
    return render_template("login.html")

@app.route('/signup', methods=['POST'])
def signup():
    print(f"Received request: {request.form}")

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if not username or not email or not password:
        return jsonify({"message": "All fields are required", "status": "error"}), 400

    # Check if the user already exists
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"message": "Username already taken", "status": "error"}), 400

    hashed_password = generate_password_hash(password)

    mongo.db.users.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password,
        "favorites": []  # optional: initialize favorites or other fields
    })

    # Set session just like in login
    session['username'] = username
    session['user_logged_in'] = True
    session['favorites'] = []

    return render_template("home.html", user_logged_in=True)

@app.route("/logout")
def logout():
    session.clear()  # Clears session data
    return redirect(url_for('home'))

@app.route('/favorite-business', methods=['POST'])
def favorite_business():
    if 'user_logged_in' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    data = request.json
    business_name = data.get('business_name')

    if not business_name:
        return jsonify({'error': 'Business name is required'}), 400

    username = session.get('username')
    user = mongo.db.users.find_one({"username": username})

    if not user:
        return jsonify({'error': 'User not found'}), 404

    favorites = user.get('favorites', [])
    
    if business_name in favorites:
        favorites.remove(business_name)
        favorited = False
    else:
        favorites.append(business_name)
        favorited = True

    mongo.db.users.update_one(
        {"username": username},
        {"$set": {"favorites": favorites}}
    )
    return jsonify({'message': 'Favorite status updated', 'favorited': favorited})

@app.route('/user-favorites', methods=['GET'])
def get_user_favorites():
    if 'user_logged_in' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session.get('username')
    user = mongo.db.users.find_one({"username": username})

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'favorites': user.get('favorites', [])})  # Return empty list if missing

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_profile_picture', methods=['POST'])
def upload_profile_picture():
    if 'username' not in session:
        return jsonify({"message": "Unauthorized", "status": "error"}), 401

    if 'profile_picture' not in request.files:
        return jsonify({"message": "No file part", "status": "error"}), 400

    file = request.files['profile_picture']

    if file.filename == '':
        return jsonify({"message": "No selected file", "status": "error"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Update the user's profile picture in the database
        username = session['username']
        mongo.db.users.update_one({"username": username}, {"$set": {"profile_picture": file_path}})

        return jsonify({"message": "Profile picture updated successfully", "status": "success", "image_url": file_path}), 200

    return jsonify({"message": "Invalid file format", "status": "error"}), 400


# Route to get all business data (limit 1000)
@app.route("/get", methods=["GET"])
def get_business_data():
    business_data = mongo.db.mycollection.find()
    return jsonify([data for data in business_data])

@app.route('/business/<business_name>/get_reviews', methods=['GET'])
def get_reviews():
    business_name = request.args.get("business_name")

    if not business_name:
        return jsonify({"message": "Business name required", "status": "error"}), 400

    reviews = list(mongo.db.mycollection.find({"business_name": business_name}, {"_id": 0}))  # Exclude MongoDB _id
    return jsonify({"reviews": reviews})

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fetch the data from MongoDB
def get_data_from_mongo():
    """Fetch business data from MongoDB and return as a DataFrame."""
    business_data = mongo.db.mycollection.find()  # Adjust collection name if needed
    # Convert the MongoDB cursor to a DataFrame
    data = pd.DataFrame(list(business_data))
    return data

def smart_search(query, tfidf_matrix, data, top_n=5):
    query_vec = vectorizer.transform([query])  # Convert the query to TF-IDF vector
    similarities = cosine_similarity(query_vec, tfidf_matrix)  # Calculate similarity
    
    # Get top N matching businesses based on similarity
    top_indices = similarities[0].argsort()[-top_n:][::-1]  # Get indices of the highest similarity
    
    # Return the top N business details as a list of dictionaries
    top_data = data.iloc[top_indices]  # This will give you a DataFrame
    return top_data.to_dict(orient='records')  # Convert to list of dictionaries

@app.route("/search", methods=["GET"])
def search_business():
    query = request.args.get("query")  # Get the query parameter
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Fetch business data from MongoDB
    # Fetch business data from MongoDB
    data = get_data_from_mongo()

    # Extract the text (reviews or descriptions) column from the data
    reviews = data['text'].astype(str).tolist()  # Adjust if your reviews are in a different column

    # If 'text' is a list of lists, flatten it
    reviews = [str(item) for sublist in reviews for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Convert reviews to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(reviews)
    
    # Perform the search using cosine similarity
    results = smart_search(query, tfidf_matrix, data)

    # Clean the text field before passing to the template (if necessary)
    for result in results:
        print(result)  # Debugging line to inspect result
        
        # Convert the 'text' field from a string to a list
        result['text'] = ast.literal_eval(result['text'])
    
    # Render the results using a template (search_results.html)
    return render_template("search_results.html", query=query, results=results)

def clean_text(text):
    # Example preprocessing steps
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Route to get top-rated recommended businesses
@app.route("/recommended", methods=["GET"])
def recommended_businesses():
    try:
        recommended = mongo.db.mycollection.find({}, {'_id': 0}).sort("avg_rating", -1).limit(5)
        return jsonify([data for data in recommended])
    except Exception as e:
        print(f"Error fetching recommended businesses: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Route to get map data (limit 100)
@app.route("/mapdata", methods=["GET"])
def map_data():
    try:
        business_data = mongo.db.mycollection.find({}, {'_id': 0}).limit(1000)
        return jsonify([data for data in business_data])
    except Exception as e:
        print(f"Error fetching map data: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/business/<business_name>')
def business_page(business_name):
    try:
        business_name = unquote(business_name)
        print(f"Looking for business: {business_name}")

        business_data = get_business_by_name(business_name)


        if not business_data:
            print(f"Business '{business_name}' not found")
            return "Business not found", 404

        return render_template('businesspage.html', business=business_data)
    except Exception as e:
        print(f"Error: {e}")
        return f"Internal Server Error: {e}", 500


def get_business_by_name(business_name):
    try:
        print(f"Fetching data for business: {business_name}")
        business = mongo.db.mycollection.find_one(
            {'name': {'$regex': business_name, '$options': 'i'}}
        )

        if business:
            print(f"Found business: {business}")
            # Handle text and rating fields properly
            if isinstance(business.get("text"), str):
                try:
                    business["text"] = ast.literal_eval(business["text"])
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing 'text' field: {e}")
                    business["text"] = []

            if isinstance(business.get("rating"), str):
                try:
                    business["rating"] = ast.literal_eval(business["rating"])
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing 'rating' field: {e}")
                    business["rating"] = []
            
            # Check if the 'category' field is a string, and if so, try to convert it into a list
            if isinstance(business.get("category"), str):
                try:
                    business["category"] = ast.literal_eval(business["category"])
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing 'category' field: {e}")
                    business["category"] = []
        else:
            print(f"Business '{business_name}' not found in database.")

        return business
    except Exception as e:
        print(f"Error fetching business data: {e}")
        return None

@app.route('/submit_review', methods=['POST'])
def submit_review():
    if 'username' not in session:
        return jsonify({"message": "Unauthorized", "status": "error"}), 401

    data = request.json
    review_text = data.get("review")
    business_name = data.get("business_name")  # Use business name instead of ID

    if not review_text or not business_name:
        return jsonify({"message": "Review text and business name are required", "status": "error"}), 400

    review = {
        "username": session["username"],
        "business_name": business_name,
        "review": review_text,
        "timestamp": datetime.utcnow()
    }

    mongo.db.reviews.insert_one(review)

    return jsonify({"message": "Review submitted successfully", "status": "success"}), 201

# Read API key from a local file
def load_api_key(filename=".openai_key"):
    try:
        with open(filename, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: API key file not found.")
        return None

api_key = load_api_key()
if api_key:
    client = openai.Client(api_key=api_key)   
else:
    raise ValueError("API key is missing. Please check your file.")

def generate_business_description(reviews_text):
    prompt = f"Based on the following customer reviews, generate a concise business description that summarizes the key offerings and nature of the business:\n\n{reviews_text}"

    response = client.chat.completions.create( 
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates business descriptions based on customer reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,  
        temperature=0.7,
    )

    description = response.choices[0].message.content.strip()
    return description

def summarize_reviews(reviews_text):
    if not reviews_text:
        return "No reviews available for summarization."

    prompt = "Summarize the following customer reviews:\n\n" + "\n".join(reviews_text)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes customer reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    print(response)
    
    # Correct way to access the message content
    summary = response.choices[0].message.content.strip()
    return summary


@app.route('/business/<business_name>/generate_description', methods=["POST"])
def generate_business_description_route(business_name):
    try:
        business_name = unquote(business_name)  # Decode the URL-encoded business name
        print(f"Generating description for business: {business_name}")

        # Fetch the business data from MongoDB
        business_data = get_business_by_name(business_name)

        if not business_data:
            return jsonify({"error": "Business not found"}), 404
        
        # Retrieve the reviews from the 'text' field
        reviews_text = business_data.get("text", "")
        if not reviews_text:
            return jsonify({"error": "No reviews available for this business"}), 404
        
        # Generate the business description using OpenAI
        description = generate_business_description(reviews_text)

        return jsonify({"description": description})

    except Exception as e:
        print(f"Error generating description: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/business/<business_name>/summarize_reviews', methods=["POST"])
def summarize_reviews_route(business_name):
    try:
        business_name = unquote(business_name)  # Decode the URL-encoded business name
        print(f"Summarizing reviews for business: {business_name}")

        # Fetch the business data from MongoDB
        business_data = get_business_by_name(business_name)

        if not business_data:
            return jsonify({"error": "Business not found"}), 404
        
        # Retrieve the reviews from the 'text' field
        reviews_text = business_data.get("text", [])
        if not reviews_text:
            return jsonify({"error": "No reviews available for this business"}), 404
        
        # Summarize the reviews using OpenAI
        summary = summarize_reviews(reviews_text)

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error summarizing reviews: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Function to chat with the bot
def chat_with_bot(user_message, reviews_text, business_data):
    if not reviews_text:
        return "No reviews available for summarization."

    # Build the prompt with the user's inquiry and available reviews
    prompt = f"Answer the user's inquiry about the business '{business_data['name']}' based on the following reviews:\n\n{reviews_text}\nUser's Question: {user_message}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # GPT-3.5 model
        messages=[
            {"role": "system", "content": "You are an assistant that responds to user inquiries using customer reviews as the basis for your answers."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,  # Adjust this as needed
        temperature=0.7,
    )
    print(response)

    # Extract the response content
    bot_response = response.choices[0].message.content.strip()

    return bot_response

# Define the /chat route to handle incoming requests from the frontend
@app.route('/business/<business_name>/chat', methods=['POST'])
def chat(business_name):
    # Get the data from the POST request
    data = request.get_json()
    business_name = unquote(business_name)  # Decode the URL-encoded business name
    print(f"Generating chat for business: {business_name}")

    # Extract user message and reviews text from the request body
    user_message = data['user_message']
    reviews_text = data['reviews_text']
    
    # Fetch business data using the decoded business name
    business_data = get_business_by_name(business_name)
    
    # Check if business data is found
    if not business_data:
        return jsonify({"error": "Business not found"}), 404

    # Call the chat_with_bot function to get the bot's response
    bot_response = chat_with_bot(user_message, reviews_text, business_data)

    # Return the bot's response as JSON
    return jsonify({'bot_response': bot_response})


if __name__ == "__main__":
    # Connect to MongoDB
    try:
        mongo.cx = mongo.cx
        print("Database is connected!")
        app.run(port=int(os.getenv("PORT", 5001)), debug=True)
    except Exception as e:
        print(f"Database connection error: {e}")

