// JavaScript

import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.css';

document.getElementById('start_text').innerText = "Hello World";

// loadGraphModel
tf.loadLayersModel('http://34.67.197.16:6100/model.json').then(model => {
  console.log('Model loaded');
  model.predict(tf.ones([1, 28, 28])).print();
});

const eg = tf.tensor([[1.0, 1.0], [1.0, 1.0]]);
eg.print();


document.getElementById('output').innerText = "prediction";
