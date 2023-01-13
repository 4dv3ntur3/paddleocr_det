import sys

from flask import Flask, request, jsonify, abort
app = Flask(__name__)

from PaddleOCR import predict_system
import textsys


def ocr_demo():
    
    # rest API
    # 
    
    return 



if __name__ == '__main__':
    app.run(debug=False)