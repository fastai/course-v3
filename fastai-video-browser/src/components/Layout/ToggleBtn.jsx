import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

import { standard } from '../../utils/easing';

const Btn = styled.button`
	height: 50px;
  width: 50px;
  background: rgba(0, 0, 0, 0.3);
	border: none;
	cursor: pointer;
	color: var(--text-light);
	border-radius: 50%;
	line-height: 0;
  z-index: 2;
	position: absolute;
  transition: background 0.3s ${standard};
	&:hover {
		background: var(--fastai-blue);
	}
	& > svg {
		width: 20px;
		height: 20px;
  }
`;

const ToggleBtn = ({ shown, ShownIcon, HiddenIcon, ...rest }) => (
  <Btn {...rest}>
    { shown ? <ShownIcon /> : <HiddenIcon /> }
  </Btn>
);

ToggleBtn.propTypes = {
  shown: PropTypes.bool.isRequired,
  ShownIcon: PropTypes.func.isRequired,
  HiddenIcon: PropTypes.func.isRequired
}

export default ToggleBtn;
