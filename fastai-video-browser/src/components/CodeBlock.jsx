import React from 'react';
import PropTypes from 'prop-types';
import SyntaxHighlighter from 'react-syntax-highlighter';

const CodeBlock = ({ language, value }) => (
  <SyntaxHighlighter language={language}>
    {value}
  </SyntaxHighlighter>
);

CodeBlock.propTypes = {
  value: PropTypes.string.isRequired,
  language: PropTypes.string,
}

CodeBlock.defaultProps = {
  language: null,
}

export default CodeBlock;