import React, { Component } from 'react';
import App from './App';
import PasswordChecker from './components/PasswordChecker';

export default class AuthWrapper extends Component {
  state = {
    authorized: true,
  };

  authorized = () => {
    this.setState({ authorized: true });
  };

  render() {
    const { authorized } = this.state;

    if (!authorized) return <PasswordChecker authorized={this.authorized} />;

    return <App />;
  }
}
