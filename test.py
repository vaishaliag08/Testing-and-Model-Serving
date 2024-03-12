import pickle
import sklearn
import score
import requests
import subprocess 
import time
import flask

svc_model = pickle.load(open("support_vector.pkl", "rb"))
text = "Subject: make big money with foreclosed real estate in your area  trinity consulting 1730 redhill ave . , ste . 135 irvine , ca 92606 this e - mail message is an advertisement and / or solicitation . "
threshold = 0.95

class TestScore:
    expected_input_format = (str, sklearn.svm._classes.SVC, float)
    expected_output_foramt = (bool, float)

    def test_smoke(self):
        try:
            output = score.score(text, svc_model, threshold)
            assert output is not None
        except Exception:
            return False
        
    def test_format(self):
        sample_input = (text, svc_model, threshold)
        sample_input_format = (type(input_) for input_ in sample_input)
        if sample_input_format == self.expected_input_format:
            try:
                output = score.score(*sample_input)
                output_format = (type(output[0]), type(output[1]))
                assert output_format == self.expected_output_foramt
            except Exception:
                return "NO"

    def test_prediction(self):
        output = score.score(text, svc_model, threshold)
        prediction = output[0]
        assert prediction in (0, 1)
    
    def test_propensity(self):
        output = score.score(text, svc_model, threshold)
        propensity = output[1]
        assert propensity >= 0 and propensity <= 1
    
    def test_threshold_0(self):
        threshold = 0
        output = score.score(text, svc_model, threshold)
        prediction = output[0]
        assert prediction == 1
    
    def test_threshold_1(self):
        threshold = 1
        output = score.score(text, svc_model, threshold)
        prediction = output[0]
        assert prediction == 0
    
    def test_spam_input(self):
        spam_input = "claim your free gift card worth $1000 on home depot. Exclusive offer, were sure you can find a use for this gift card."
        output = score.score(spam_input, svc_model, threshold)
        prediction = output[0]
        assert prediction == 1
    
    def test_non_spam_input(self):
        non_spam_input = "The meeting is scheduled for tomoroow 5 PM with the project manager. Be there on time."
        output = score.score(non_spam_input, svc_model, threshold)
        prediction = output[0]
        assert prediction == 0


def test_flask():
    # Launch the Flask app
    flask_process = subprocess.Popen(["python3", "app.py"])

    time.sleep(2)

    try:
        # Test the response from the localhost endpoint
        payload = {
            "text": "Hurray! You won a jackpot for $1000. Get the exclusive discount now. Don't miss out the opportunity!",
            "threshold": 0.95
        }
        response = requests.post('http://127.0.0.1:5000', data=payload)
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'propensity' in data
    finally:
        # Close the Flask app
        flask_process.kill()
      
