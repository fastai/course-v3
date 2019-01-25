import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const StyledSearch = styled.div`
  height: 1.5rem;
  z-index: 3;
  position: absolute;
  bottom: 8px;
  left: 5px;
  font-size: 0.875rem;
  border: 1px solid black;
  input {
    padding-left: 5px;
    height: 1.5rem;
    width: 8rem;
  }
`

const Search = ({ search, handleChange }) => (
  <StyledSearch>
    <input
      value={search}
      onChange={handleChange}
      placeholder="Search transcript"
    />
  </StyledSearch>
);

Search.defaultProps = {
  search: '',
};

Search.propTypes = {
  search: PropTypes.string,
  handleChange: PropTypes.func.isRequired,
};

export default Search;
