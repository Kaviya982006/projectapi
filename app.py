from flask import Flask, request, render_template
import instaloader
import pickle
import xgboost as xgb
import sklearn
#print("Using scikit-learn version:", sklearn.__version__)#


app = Flask(__name__)

# Load your trained XGBoost model (JSON format)
model = xgb.XGBClassifier()
model.load_model("model.json")

# Load your scaler (pickle format)
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form["username"]

        try:
            loader = instaloader.Instaloader()
            profile = instaloader.Profile.from_username(loader.context, username)

            # Extract features (followers, following, posts, private/public)
            features = [
                profile.followers,
                profile.followees,
                profile.mediacount,
                int(profile.is_private)
            ]

            # Scale + predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]

            if prediction == 1:
                result = "Fake Account ❌"
                # Cybercrime portal link (user must manually file complaint)
                complaint_link = f"https://cybercrime.gov.in/Webform/complaint.aspx?username={username}"
            else:
                result = "Real Account ✅"
                complaint_link = None

            return render_template("result.html", username=username, result=result, complaint_link=complaint_link)

        except Exception as e:
            return f"Error fetching profile: {str(e)}"

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)

