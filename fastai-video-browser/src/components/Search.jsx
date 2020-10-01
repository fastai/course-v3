import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const SearchContainer = styled.div`
  width: 100%;
  height: 90px;
  display: flex;
  align-items: stretch;
  input {
    background: #efefef;
    padding: 12px 24px;
    flex: 1;
    font-size: 2.5rem;
    border: none;
    border-bottom: 1px solid #eee;
    &::placeholder {
      color: #777;
      font-weight: normal;
      font-family: 'Karla', sans-serif;
    }
    &[disabled]::placeholder {
      color: transparent;
    }
  }
`

const Search = ({ search, handleChange, ...rest }) => (
  <SearchContainer>
    <input
      value={search}
      onChange={handleChange}
      placeholder="Search transcript"
      {...rest}
    />
  </SearchContainer>
);

Search.defaultProps = {
  search: '',
};

Search.propTypes = {
  search: PropTypes.string,
  handleChange: PropTypes.func.isRequired,
};

export default Search;
