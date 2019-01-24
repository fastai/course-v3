import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const StyledSearch = styled.div`
  flex: 8;
  margin-right: 2%;
  input {
    padding-left: 5px;
    margin-left: 5px;
    height: 1.5rem;
    min-width: 11vw;
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
