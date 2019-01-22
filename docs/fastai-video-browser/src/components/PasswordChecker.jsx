import React, { Component } from 'react';
import PropTypes from 'prop-types';
import crypto from 'crypto-js';
import styled from 'styled-components';
import { Col, Container, Hidden, Row, Visible } from 'react-grid-system';

import BrandLogo from './BrandLogo';

import HASH from '../utils/hash';

const inputStyles = `
  height        : 40px;
  width         : 100%;
  margin-top    : 10px;
  border-radius : 3px;
  font-size     : 20px;
  line-height   : 36px;
  text-align    : center;
`;

const StyledInput = styled.input`
  border-radius: 3px;
  ${inputStyles}
`;

const StyledButton = styled.button`
  cursor: pointer;
  ${inputStyles}
`;

class PasswordChecker extends Component {
  state = {
    plaintext: '',
  }

  checkPass = (e) => {
    const { authorized } = this.props;
    const { plaintext } = this.state;

    e.preventDefault();
    e.stopPropagation();

    const hashed = crypto.SHA3(plaintext).toString();
    // console.log(hashed, hashed === HASH);
    if (hashed === HASH) return authorized();
    this.setState({ plaintext: '' });
    return null;
  }

  handleChange = (e) => {
    this.setState({
      plaintext: e.target.value,
    });
  }

  render() {
    const { plaintext } = this.state;

    return (
      <form onSubmit={this.checkPass}>
        <Container>
          {/* center content vertically */}
          <Row align="center" style={{ height: '100vh' }}>
            <Col>
              <Row>
                <Col xs={12} sm={4} lg={3}>
                  <Visible xs>
                    <BrandLogo />
                  </Visible>
                  <Hidden xs>
                    <BrandLogo right />
                  </Hidden>
                </Col>
                <Col xs={12} sm={5}>
                  <StyledInput
                    type="password"
                    value={plaintext}
                    onChange={this.handleChange}
                  />
                </Col>
                <Col xs={12} sm={3}>
                  <StyledButton type="submit">Log In</StyledButton>
                </Col>
              </Row>
            </Col>
          </Row>
        </Container>
      </form>
    );
  }
}

PasswordChecker.propTypes = {
  authorized: PropTypes.func.isRequired,
};

export default PasswordChecker;
