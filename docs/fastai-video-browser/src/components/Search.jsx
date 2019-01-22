import React from 'react';
import PropTypes from 'prop-types';

const Search = ({ search, handleChange }) => (
  <div className="Search">
    <input
      className="fl w-100"
      value={search}
      onChange={handleChange}
      placeholder="search transcripts and chapter headings..."
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
