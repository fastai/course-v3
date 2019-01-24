import React from 'react';
import PropTypes from 'prop-types';

const Search = ({ search, handleChange }) => (
  <div className="Search">
    <input
      value={search}
      onChange={handleChange}
      placeholder="Search transcript"
    />
  </div>
);

Search.defaultProps = {
  search: '',
};

Search.propTypes = {
  search: PropTypes.string,
  handleChange: PropTypes.func.isRequired,
};

export default Search;
