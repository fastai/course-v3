import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';

function BrandLogo(props) {
  const { right } = props;

  const StyledSiteLogo = styled.div`
    text-align      : center;
    font-family     : 'Abril Fatface', serif;
    font-weight     : 700;
    text-decoration : underline;
    color           : white;
    font-size       : 55px;
    line-height     : 55px;
    text-align      : ${right ? 'right' : 'center'}
  `;

  return (
    <StyledSiteLogo>
      <a
        href="http://fast.ai"
        target="_blank"
        rel="noopener noreferrer"
      >
        fast.ai
      </a>
    </StyledSiteLogo>
  );
}

BrandLogo.defaultProps = {
  right: false,
};

BrandLogo.propTypes = {
  right: PropTypes.bool,
};

export default BrandLogo;
