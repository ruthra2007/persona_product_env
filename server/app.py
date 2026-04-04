from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "OpenEnv Server Running ✅"

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({
        "observation": {},
        "reward": 1.0,
        "done": True,
        "info": {}
    })