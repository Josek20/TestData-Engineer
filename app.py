from flask import Flask, render_template, request

from main import pipeline_for_new_input
from init_phrase_manager import phrase_manager

app = Flask(__name__)


@app.route('/',  methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # Check if the request method is POST
        input_phrase = request.form.get('phrase')  # Use request.form to get form data
        if input_phrase:
            closest_phrase, l2_distance, cos_distance = pipeline_for_new_input(input_phrase, phrase_manager)
            return render_template('index.html', response=[closest_phrase, l2_distance, cos_distance])
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

