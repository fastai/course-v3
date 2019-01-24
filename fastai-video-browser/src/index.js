/* eslint-disable react/jsx-filename-extension */

import React from 'react';
import ReactDOM from 'react-dom';
import { Router } from 'react-router-dom';
import createHistory from 'history/createBrowserHistory';
import 'normalize.css';
import App from './App';

const history = createHistory()

ReactDOM.render(<Router history={history}><App /></Router>, document.getElementById('root'));
