import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

import { standard } from '../../utils/easing';

const PanelWrapper = styled.aside`
	background: var(--background);
	box-shadow: 2px 0 30px rgba(128, 128, 128, 0.3);
	transition: all 0.3s ${standard};
	z-index: 1;
	flex-shrink: 0;
	overflow-x: hidden;
	overflow-y: scroll;
	width: ${ props => props.width};
	${ props => !props.shown && `
			width: 0px;
 `}
`;


// adding a div with set width for the inner content prevents reflow on show/hide
const Panel = ({ children, shown, width, ...rest }) => ( 
	<PanelWrapper shown={shown} width={width} {...rest}>
		<div style={{ width, height: '100%' }}>
			{ children }
		</div>
	</PanelWrapper>
);

Panel.propTypes = {
	shown: PropTypes.bool.isRequired,
	width: PropTypes.string
};

Panel.defaultProps = {
	width: '250px'
}

export default Panel;
