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

# 🔥 REQUIRED MAIN FUNCTION
def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()