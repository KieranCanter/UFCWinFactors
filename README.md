<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>finalproject</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0px solid transparent;
  border-right: 0px solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0px solid transparent;
  border-bottom: 0px solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
	border:1px solid transparent;
  background-color: transparent;
  position: absolute;
	z-index:1;
	right:3%;
	top: 0;
	bottom: 0;
	margin: auto;
	padding: 7px 0;
	display: none;
	vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
	content: "X";
	display: block;
	width: 15px;
	height: 15px;
	text-align: center;
	color:#000;
	font-weight: normal;
	font-size: 12px;
	cursor: pointer;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
  background: inherit;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-bottom:10px;
  margin-top:0; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  font-size:36px;
  line-height:40px; }

h2.bp3-heading, .bp3-running-text h2{
  font-size:28px;
  line-height:32px; }

h3.bp3-heading, .bp3-running-text h3{
  font-size:22px;
  line-height:25px; }

h4.bp3-heading, .bp3-running-text h4{
  font-size:18px;
  line-height:21px; }

h5.bp3-heading, .bp3-running-text h5{
  font-size:16px;
  line-height:19px; }

h6.bp3-heading, .bp3-running-text h6{
  font-size:14px;
  line-height:16px; }
.bp3-ui-text{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none; }

.bp3-monospace-text{
  font-family:monospace;
  text-transform:none; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  font-size:14px;
  line-height:1.5; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    margin:20px 0; }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  color:#106ba3;
  text-decoration:none; }
  a:hover{
    color:#106ba3;
    cursor:pointer;
    text-decoration:underline; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  font-size:smaller;
  padding:2px 5px; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  color:#182026;
  display:block;
  font-size:13px;
  line-height:1.4;
  margin:10px 0;
  padding:13px 15px 12px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit;
    font-size:inherit;
    padding:0; }

.bp3-running-text kbd, .bp3-key{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-family:inherit;
  font-size:12px;
  height:24px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  line-height:24px;
  min-width:24px;
  padding:3px 6px;
  vertical-align:middle; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  margin:0 0 10px;
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    font-size:40px;
    margin-right:20px;
    margin-top:0; }

.bp3-alert-contents{
  word-break:break-word; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  cursor:default;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  height:30px;
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-breadcrumbs > li{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-breadcrumbs > li::after{
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      content:"";
      display:block;
      height:16px;
      margin:0 5px;
      width:16px; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    font-size:inherit;
    font-weight:inherit;
    vertical-align:baseline; }

.bp3-breadcrumbs-collapsed{
  background:#ced9e0;
  border:none;
  border-radius:3px;
  cursor:pointer;
  margin-right:2px;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    content:"";
    display:block;
    height:16px;
    width:16px; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    color:#182026;
    text-decoration:none; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  min-height:30px;
  min-width:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      background-color:#0e5a8a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      background-color:#0d8050;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      background-color:#0a6640;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      background-color:#bf7326;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      background-color:#a66321;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      background-color:#c23030;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      background-color:#a82a2a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-height:40px;
    min-width:40px;
    font-size:16px;
    padding:5px 15px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      margin:0;
      position:absolute; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button.bp3-minimal:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button.bp3-outlined{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    border:1px solid rgba(24, 32, 38, 0.2);
    -webkit-box-sizing:border-box;
            box-sizing:border-box; }
    .bp3-button.bp3-outlined:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-outlined:active, .bp3-button.bp3-outlined.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-outlined{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-outlined:hover, .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-outlined:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover, .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover, .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover, .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover, .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled:hover{
      border-color:rgba(92, 112, 128, 0.1); }
    .bp3-dark .bp3-button.bp3-outlined{
      border-color:rgba(255, 255, 255, 0.4); }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        border-color:rgba(255, 255, 255, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      border-color:rgba(16, 107, 163, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        border-color:rgba(16, 107, 163, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        border-color:rgba(72, 175, 240, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          border-color:rgba(72, 175, 240, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      border-color:rgba(13, 128, 80, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        border-color:rgba(13, 128, 80, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        border-color:rgba(61, 204, 145, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          border-color:rgba(61, 204, 145, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      border-color:rgba(191, 115, 38, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        border-color:rgba(191, 115, 38, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        border-color:rgba(255, 179, 102, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          border-color:rgba(255, 179, 102, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      border-color:rgba(194, 48, 48, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        border-color:rgba(194, 48, 48, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        border-color:rgba(255, 115, 115, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          border-color:rgba(255, 115, 115, 0.2); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    border-bottom-right-radius:0;
    border-top-right-radius:0;
    margin-right:-1px; }
  .bp3-button-group.bp3-minimal .bp3-button{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      height:100%;
      width:unset; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  font-size:14px;
  line-height:1.5;
  background-color:rgba(138, 155, 168, 0.15);
  border-radius:3px;
  padding:10px 12px 9px;
  position:relative;
  width:100%; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout .bp3-heading{
    line-height:20px;
    margin-bottom:5px;
    margin-top:0; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  opacity:0.9;
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  width:100%; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }

.bp3-dialog{
  background:#ebf1f5;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text;
  width:500px; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    background:#293742;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-dialog-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding-left:20px;
  padding-right:5px;
  z-index:30; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    background:#30404d;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  margin:20px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-multistep-dialog-panels{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }

.bp3-multistep-dialog-left-panel{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column; }
  .bp3-dark .bp3-multistep-dialog-left-panel{
    background:#202b33; }

.bp3-multistep-dialog-right-panel{
  background-color:#f5f8fa;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  border-radius:0 0 6px 0;
  -webkit-box-flex:3;
      -ms-flex:3;
          flex:3;
  min-width:0; }
  .bp3-dark .bp3-multistep-dialog-right-panel{
    background-color:#293742;
    border-left:1px solid rgba(16, 22, 26, 0.4); }

.bp3-multistep-dialog-footer{
  background-color:#ffffff;
  border-radius:0 0 6px 0;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  padding:10px; }
  .bp3-dark .bp3-multistep-dialog-footer{
    background:#30404d;
    border-top:1px solid rgba(16, 22, 26, 0.4); }

.bp3-dialog-step-container{
  background-color:#f5f8fa;
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-dialog-step-container{
    background:#293742;
    border-bottom:1px solid rgba(16, 22, 26, 0.4); }
  .bp3-dialog-step-container.bp3-dialog-step-viewed{
    background-color:#ffffff; }
    .bp3-dark .bp3-dialog-step-container.bp3-dialog-step-viewed{
      background:#30404d; }

.bp3-dialog-step{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#f5f8fa;
  border-radius:6px;
  cursor:not-allowed;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:4px;
  padding:6px 14px; }
  .bp3-dark .bp3-dialog-step{
    background:#293742; }
  .bp3-dialog-step-viewed .bp3-dialog-step{
    background-color:#ffffff;
    cursor:pointer; }
    .bp3-dark .bp3-dialog-step-viewed .bp3-dialog-step{
      background:#30404d; }
  .bp3-dialog-step:hover{
    background-color:#f5f8fa; }
    .bp3-dark .bp3-dialog-step:hover{
      background:#293742; }

.bp3-dialog-step-icon{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:rgba(92, 112, 128, 0.6);
  border-radius:50%;
  color:#ffffff;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:25px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:25px; }
  .bp3-dark .bp3-dialog-step-icon{
    background-color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#2b95d6; }
  .bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#8a9ba8; }

.bp3-dialog-step-title{
  color:rgba(92, 112, 128, 0.6);
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  padding-left:10px; }
  .bp3-dark .bp3-dialog-step-title{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-title{
    color:#2b95d6; }
  .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
    color:#182026; }
    .bp3-dark .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
      color:#f5f8fa; }
.bp3-drawer{
  background:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    height:50%;
    left:0;
    right:0;
    top:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-bottom{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-left{
    bottom:0;
    left:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-right{
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    background:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-drawer-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding:5px;
  padding-left:20px;
  position:relative; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  overflow:auto; }

.bp3-drawer-footer{
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  padding:10px 20px;
  position:relative; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  cursor:text;
  display:inline-block;
  max-width:100%;
  position:relative;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    bottom:-3px;
    left:-3px;
    position:absolute;
    right:-3px;
    top:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  color:inherit;
  display:inherit;
  font:inherit;
  letter-spacing:inherit;
  max-width:inherit;
  min-width:inherit;
  position:relative;
  resize:none;
  text-transform:inherit;
  vertical-align:top; }

.bp3-editable-text-input{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0;
  white-space:pre-wrap;
  width:100%; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    left:0;
    position:absolute;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    border-radius:inherit;
    z-index:2; }
    .bp3-control-group .bp3-input:focus{
      border-radius:3px;
      z-index:14; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    border-radius:inherit;
    z-index:4; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-left-container,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group .bp3-select:focus-within{
    z-index:5; }
  .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:-1px; }
  .bp3-control-group:not(.bp3-vertical) > .bp3-divider:not(:first-child){
    margin-left:6px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    border-radius:0 3px 3px 0;
    margin-right:0; }
  .bp3-control-group > :only-child{
    border-radius:3px;
    margin-right:0; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group .bp3-numeric-input:not(:first-child) .bp3-input-group{
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-control-group.bp3-fill{
    width:100%; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      border-radius:3px 3px 0 0;
      margin-top:0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  cursor:pointer;
  display:block;
  margin-bottom:10px;
  position:relative;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    left:0;
    opacity:0;
    position:absolute;
    top:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    cursor:pointer;
    display:inline-block;
    font-size:16px;
    height:1em;
    margin-right:10px;
    margin-top:-3px;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    vertical-align:middle;
    width:1em; }
    .bp3-control .bp3-control-indicator::before{
      content:"";
      display:block;
      height:1em;
      width:1em; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    background:#d8e1e8;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-left:10px;
    margin-top:1px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:auto; }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      background:#ffffff;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      height:calc(1em - 4px);
      left:0;
      margin:2px;
      position:absolute;
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:calc(1em - 4px); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    font-size:0.7em;
    text-align:center; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    line-height:0;
    margin-left:0.5em;
    margin-right:1.2em;
    visibility:hidden; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    line-height:1em;
    margin-left:1.2em;
    margin-right:0.5em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    line-height:1em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    line-height:0;
    visibility:hidden; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      background:#202b33;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  cursor:pointer;
  display:inline-block;
  height:30px;
  position:relative; }
  .bp3-file-input input{
    margin:0;
    min-width:200px;
    opacity:0; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:rgba(92, 112, 128, 0.6);
  left:0;
  padding-right:80px;
  position:absolute;
  right:0;
  top:0;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-file-upload-input::after{
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    min-height:24px;
    min-width:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    border-radius:3px;
    content:"Browse";
    line-height:24px;
    margin:3px;
    position:absolute;
    right:0;
    text-align:center;
    top:0;
    width:70px; }
    .bp3-file-upload-input::after:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-file-upload-input:active::after{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-large .bp3-file-upload-input{
    font-size:16px;
    height:40px;
    line-height:40px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-height:30px;
      min-width:30px;
      line-height:30px;
      margin:5px;
      width:85px; }
  .bp3-dark .bp3-file-upload-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:active::after{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    color:#5c7080;
    font-size:12px;
    margin-top:5px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      line-height:40px;
      margin:0 10px 0 0; }
    .bp3-form-group.bp3-inline label.bp3-label{
      line-height:30px;
      margin:0 10px 0 0; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-input-left-container:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-input-left-container:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-height:24px;
    min-width:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-icon{
    z-index:1; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon{
    color:#5c7080; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon:empty,
    .bp3-input-group > .bp3-icon:empty{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-height:30px;
    min-width:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle; }
  .bp3-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-input.bp3-large{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-bottom:15px;
  margin-top:0; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    font-weight:400;
    vertical-align:top;
    width:100%; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  min-height:0;
  padding:0;
  width:30px; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  -moz-appearance:none;
  -webkit-appearance:none;
  border-radius:3px;
  height:30px;
  padding:0 25px 0 10px;
  width:100%; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  background:none;
  -webkit-box-shadow:none;
          box-shadow:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    background:rgba(167, 182, 194, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026;
    text-decoration:none; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    background:rgba(115, 134, 148, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      color:rgba(167, 182, 194, 0.6);
      cursor:not-allowed; }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  font-size:16px;
  height:40px;
  padding-right:35px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    background-color:#202b33;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  background-color:rgba(206, 217, 224, 0.5);
  -webkit-box-shadow:none;
          box-shadow:none;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  color:#5c7080;
  pointer-events:none;
  position:absolute;
  right:7px;
  top:7px; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  letter-spacing:normal;
  position:relative;
  vertical-align:middle; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    right:12px;
    top:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select option:disabled, .bp3-dark
  .bp3-select option:disabled{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    text-align:left;
    vertical-align:top; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td,
  .bp3-running-text table tfoot tr:first-child th,
  table.bp3-html-table tfoot tr:first-child th,
  .bp3-running-text table tfoot tr:first-child td,
  table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td,
  .bp3-dark .bp3-running-text table tfoot tr:first-child th,
  .bp3-running-text .bp3-dark table tfoot tr:first-child th,
  .bp3-dark table.bp3-html-table tfoot tr:first-child th,
  .bp3-dark .bp3-running-text table tfoot tr:first-child td,
  .bp3-running-text .bp3-dark table tfoot tr:first-child td,
  .bp3-dark table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-bottom:6px;
  padding-top:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td,
table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
  table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table{ }
  .bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
    background:rgba(92, 112, 128, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td,
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
      -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
    background-color:rgba(92, 112, 128, 0.3);
    cursor:pointer; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
    background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  padding-bottom:0;
  top:40px; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-left:0;
  margin-right:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  font-family:"Icons20";
  font-size:inherit;
  font-style:normal;
  font-weight:400;
  line-height:1; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagnosis::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-lab-test::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }
  .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{ }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  background:#ffffff;
  border-radius:3px;
  color:#182026;
  list-style:none;
  margin:0;
  min-width:180px;
  padding:5px;
  text-align:left; }

.bp3-menu-divider{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px; }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  color:inherit;
  line-height:20px;
  padding:5px 7px;
  text-decoration:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    color:#5c7080;
    margin-top:2px; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit !important;
    color:rgba(92, 112, 128, 0.6) !important;
    cursor:not-allowed !important;
    outline:none !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    font-size:16px;
    line-height:22px;
    padding:9px 7px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-right:10px;
      margin-top:1px; }

button.bp3-menu-item{
  background:none;
  border:none;
  text-align:left;
  width:100%; }
.bp3-menu-header{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px;
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    line-height:17px;
    margin:0;
    padding:10px 7px 0 1px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    font-size:18px;
    padding-bottom:5px;
    padding-top:15px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item{ }
  .bp3-dark .bp3-menu-item.bp3-intent-primary{
    color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-success{
    color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning{
    color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger{
    color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item::before,
  .bp3-dark .bp3-menu-item > .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item .bp3-menu-item-label{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
    background-color:rgba(138, 155, 168, 0.3); }
  .bp3-dark .bp3-menu-item.bp3-disabled{
    color:rgba(167, 182, 194, 0.6) !important; }
    .bp3-dark .bp3-menu-item.bp3-disabled::before,
    .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  background-color:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  height:50px;
  padding:0 15px;
  position:relative;
  width:100%;
  z-index:10; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    left:0;
    position:fixed;
    right:0;
    top:0; }

.bp3-navbar-heading{
  font-size:16px;
  margin-right:15px; }

.bp3-navbar-group{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px;
  margin:0 10px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:100%;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  text-align:center;
  width:100%; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  bottom:0;
  left:0;
  position:static;
  right:0;
  top:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    overflow:hidden;
    position:fixed; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    overflow:auto;
    position:fixed; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  bottom:0;
  left:0;
  position:fixed;
  right:0;
  top:0;
  opacity:1;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  z-index:20; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }
  .bp3-panel-stack-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-panel-stack2{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack2-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack2-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack2-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack2-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack2-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack2-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-view{
    background-color:#30404d; }
  .bp3-panel-stack2-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter, .bp3-panel-stack2-push .bp3-panel-stack2-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter-active, .bp3-panel-stack2-push .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter, .bp3-panel-stack2-pop .bp3-panel-stack2-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter-active, .bp3-panel-stack2-pop .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  border-radius:3px;
  display:inline-block;
  z-index:20; }
  .bp3-popover .bp3-popover-arrow{
    height:30px;
    position:absolute;
    width:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      height:20px;
      margin:5px;
      width:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-bottom:17px;
    margin-top:-17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-left:-17px;
    margin-right:17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover .bp3-popover-content{
    border-radius:3px;
    position:relative; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  border-radius:2px;
  content:"";
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg); }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  left:0;
  position:absolute;
  right:0;
  top:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  background:rgba(92, 112, 128, 0.2);
  border-radius:40px;
  display:block;
  height:8px;
  overflow:hidden;
  position:relative;
  width:100%; }
  .bp3-progress-bar .bp3-progress-meter{
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    border-radius:40px;
    height:100%;
    position:absolute;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:100%; }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  color:transparent !important;
  cursor:default;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  height:40px;
  min-width:150px;
  width:100%;
  cursor:default;
  outline:none;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    cursor:not-allowed;
    opacity:0.5; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  height:6px;
  left:0;
  right:0;
  top:5px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  height:16px;
  left:0;
  position:absolute;
  top:0;
  width:16px; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab;
    z-index:2; }
  .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    background:#bfccd6;
    -webkit-box-shadow:none;
            box-shadow:none;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    background:#5c7080;
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-slider-handle .bp3-slider-label{
    background:#394b59;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    color:#f5f8fa;
    margin-left:8px; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      background:#e1e8ed;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-bottom-right-radius:0;
    border-top-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    border-bottom-left-radius:0;
    border-top-left-radius:0;
    margin-left:8px; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  font-size:12px;
  line-height:1;
  padding:2px 5px;
  position:absolute;
  vertical-align:top; }

.bp3-slider.bp3-vertical{
  height:150px;
  min-width:40px;
  width:40px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    bottom:0;
    height:auto;
    left:5px;
    top:0;
    width:6px; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-left:0;
      margin-top:-8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      height:8px;
      margin-left:0;
      width:16px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-bottom-right-radius:3px;
      border-top-left-radius:0; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      border-bottom-left-radius:0;
      border-bottom-right-radius:0;
      border-top-left-radius:3px;
      margin-bottom:8px; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round;
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      padding:0 10px;
      width:100%; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        background-color:rgba(19, 124, 189, 0.2);
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      background-color:rgba(19, 124, 189, 0.2);
      border-radius:3px;
      bottom:0;
      height:auto;
      left:0;
      right:0;
      top:0; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  border:none;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  list-style:none;
  margin:0;
  padding:0;
  position:relative; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:#182026;
  cursor:pointer;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  font-size:14px;
  line-height:30px;
  max-width:100%;
  position:relative;
  vertical-align:top; }
  .bp3-tab a{
    color:inherit;
    display:block;
    text-decoration:none; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    background-color:transparent !important;
    -webkit-box-shadow:none !important;
            box-shadow:none !important; }
  .bp3-tab[aria-disabled="true"]{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    font-size:16px;
    line-height:40px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  left:0;
  pointer-events:none;
  position:absolute;
  top:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    background-color:#106ba3;
    bottom:0;
    height:3px;
    left:0;
    position:absolute;
    right:0; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#5c7080;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  color:#f5f8fa;
  font-size:12px;
  line-height:16px;
  max-width:100%;
  min-height:20px;
  min-width:20px;
  padding:2px 6px;
  position:relative; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-left:8px;
    padding-right:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    font-size:14px;
    line-height:20px;
    min-height:30px;
    min-width:30px;
    padding:5px 10px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-left:12px;
      padding-right:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  background:none;
  border:none;
  color:inherit;
  cursor:pointer;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin-bottom:-2px;
  margin-right:-6px !important;
  margin-top:-2px;
  opacity:0.5;
  padding:2px;
  padding-left:0; }
  .bp3-tag-remove:hover{
    background:none;
    opacity:0.8;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:0 5px 0 0; }
    .bp3-large .bp3-tag-remove:empty::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  line-height:inherit;
  min-height:30px;
  padding-left:5px;
  padding-right:0; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    color:#5c7080;
    margin-left:2px;
    margin-right:7px;
    margin-top:7px; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    margin-right:7px;
    margin-top:5px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:20px;
    width:80px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-left:5px;
      margin-top:10px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-height:30px;
      min-width:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:20px 0 0;
  max-width:500px;
  min-width:300px;
  pointer-events:all;
  position:relative !important; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-delay:50ms;
            transition-delay:50ms;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    color:#5c7080;
    margin:12px;
    margin-right:0; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    background-color:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  left:0;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none;
  right:0;
  z-index:40; }
  .bp3-toast-container.bp3-toast-container-in-portal{
    position:fixed; }
  .bp3-toast-container.bp3-toast-container-inline{
    position:absolute; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0; }
  .bp3-toast-container.bp3-toast-container-bottom{
    bottom:0;
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-exit-active ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    height:22px;
    position:absolute;
    width:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      height:14px;
      margin:4px;
      width:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-bottom:11px;
    margin-top:-11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-left:-11px;
    margin-right:11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  list-style:none;
  margin:0;
  padding-left:0; }

.bp3-tree-root{
  background-color:transparent;
  cursor:default;
  padding-left:0;
  position:relative; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:30px;
  padding-right:5px;
  width:100%; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  cursor:pointer;
  padding:7px;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  margin-right:7px;
  position:relative; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  left:calc(50% - 250px);
  top:20vh;
  width:500px;
  z-index:21; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar .bp3-input{
    background-color:transparent;
    border-radius:0; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    background-color:transparent;
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MC45NzggNTAuOTc4IiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCA1MC45NzggNTAuOTc4OyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+Cgk8Zz4KCQk8cGF0aCBzdHlsZT0iZmlsbDojMDEwMDAyOyIgZD0iTTQzLjUyLDcuNDU4QzM4LjcxMSwyLjY0OCwzMi4zMDcsMCwyNS40ODksMEMxOC42NywwLDEyLjI2NiwyLjY0OCw3LjQ1OCw3LjQ1OAoJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDAKCQkJYzYuODE2LDAsMTMuMjIxLTIuNjQ4LDE4LjAyOS03LjQ1OGM0LjgwOS00LjgwOSw3LjQ1Ny0xMS4yMTIsNy40NTctMTguMDNDNTAuOTc3LDE4LjY3LDQ4LjMyOCwxMi4yNjYsNDMuNTIsNy40NTh6CgkJCSBNNDIuMTA2LDQyLjEwNWMtNC40MzIsNC40MzEtMTAuMzMyLDYuODcyLTE2LjYxNSw2Ljg3MmgtMC4wMDJjLTYuMjg1LTAuMDAxLTEyLjE4Ny0yLjQ0MS0xNi42MTctNi44NzIKCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzIKCQkJYzQuNDMxLDQuNDMxLDYuODcxLDEwLjMzMiw2Ljg3MSwxNi42MTdDNDguOTc3LDMxLjc3Miw0Ni41MzYsMzcuNjc1LDQyLjEwNiw0Mi4xMDV6Ii8+CgkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik0yMy41NzgsMzIuMjE4Yy0wLjAyMy0xLjczNCwwLjE0My0zLjA1OSwwLjQ5Ni0zLjk3MmMwLjM1My0wLjkxMywxLjExLTEuOTk3LDIuMjcyLTMuMjUzCgkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUKCQkJYzAtMS4wOTYtMC4yNi0yLjA4OC0wLjc3OS0yLjk3OWMtMC41NjUtMC44NzktMS41MDEtMS4zMzYtMi44MDYtMS4zNjljLTEuODAyLDAuMDU3LTIuOTg1LDAuNjY3LTMuNTUsMS44MzIKCQkJYy0wLjMwMSwwLjUzNS0wLjUwMywxLjE0MS0wLjYwNywxLjgxNGMtMC4xMzksMC43MDctMC4yMDcsMS40MzItMC4yMDcsMi4xNzRoLTIuOTM3Yy0wLjA5MS0yLjIwOCwwLjQwNy00LjExNCwxLjQ5My01LjcxOQoJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQoJCQljMCwxLjE0Mi0wLjEzNywyLjExMS0wLjQxLDIuOTExYy0wLjMwOSwwLjg0NS0wLjczMSwxLjU5My0xLjI2OCwyLjI0M2MtMC40OTIsMC42NS0xLjA2OCwxLjMxOC0xLjczLDIuMDAyCgkJCWMtMC42NSwwLjY5Ny0xLjMxMywxLjQ3OS0xLjk4NywyLjM0NmMtMC4yMzksMC4zNzctMC40MjksMC43NzctMC41NjUsMS4xOTljLTAuMTYsMC45NTktMC4yMTcsMS45NTEtMC4xNzEsMi45NzkKCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}
.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}
.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}
.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-border-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0px;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-warn-color0);
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px 5px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FileDialog-Checkbox {
  margin-top: 35px;
  display: flex;
  flex-direction: row;
  align-items: end;
  width: 100%;
}

.jp-FileDialog-Checkbox > label {
  flex: 1 1 auto;
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
  overflow-x: auto;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 50px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -50px; margin-right: -50px;
  padding-bottom: 50px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 50px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
  outline: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -50px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .remote-caret {
  position: relative;
  border-left: 2px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .remote-caret > div {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -2px;
  font-size: 0.95em;
  background-color: rgb(250, 129, 0);
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 3;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .remote-caret.hide-name > div {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .remote-caret:hover > div {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
  margin: 8px 12px 0px 12px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: flex-start;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0px;
  padding-right: 2px;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 40px;
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent {
  width: 72px;
  background: var(--jp-brand-color1);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent:focus-visible {
  background-color: var(--jp-brand-color0);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent
  .jp-icon3 {
  fill: white;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileBrowser-filterBox {
  padding: 0px;
  flex: 0 0 auto;
  margin: 8px 12px 0px 12px;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing:focus-visible {
  border: 1px solid var(--jp-brand-color1);
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon:before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

body[data-format='mobile'] .jp-OutputArea-child {
  flex-direction: column;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-OutputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  margin-left: var(--jp-notebook-padding);
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

body[data-format='mobile'] .jp-InputArea {
  flex-direction: column;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
  overflow: hidden;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

body[data-format='mobile'] .jp-InputArea-editor {
  margin-left: var(--jp-notebook-padding);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-InputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

.jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
}

.jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-collapseHeadingButton {
  display: none;
}

.jp-MarkdownCell:hover .jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook-render * {
  contain: none !important;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
  float: left;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt:before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body div {
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

.highlight  {
  margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.CodeMirror pre {
  margin: 0;
  padding: 0;
}

/* Using table instead of flexbox so that we can use break-inside property */
/* CSS rules under this comment should not be required anymore after we move to the JupyterLab 4.0 CSS */


.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  min-width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-OutputArea-child {
  display: table;
  width: 100%;
}

.jp-OutputPrompt {
  display: table-cell;
  vertical-align: top;
  min-width: var(--jp-cell-prompt-width);
}

body[data-format='mobile'] .jp-OutputPrompt {
  display: table-row;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  display: table-row;
}

.jp-OutputArea-output.jp-OutputArea-executeResult {
  width: 100%;
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}
</style>

<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Analyzing-UFC-Data-and-Statistics">Analyzing UFC Data and Statistics<a class="anchor-link" href="#Analyzing-UFC-Data-and-Statistics">&#182;</a></h1>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>MMA is a rapidly growing sport in which participants, referred to as fighters, utilize various martial arts such as jiu jitsu, muay thai, kickboxing, wrestling, judo, karate, taw kwon do, and many more in an attempt to beat their opponent either by (unanimous, split, or majority) decision (UD/SD/MD), knockout/technical knockout (KO/TKO), or submission (SUB). Decisions are decided when a finish (KO/TKO or SUB) doesn't occur through the length of the fight in which a winner is chosen based on the scorecards of a panel of three judges. A unanimous decision, as per the name, is when all three judges agree on the winner, a split decision is when two out of the three judges agree and the third scored the fight for the losing fighter, and a majority decision is when two judges agree on a winner and the third scored it a draw. Although rare, a split draw can occur in a fight where all three judges scored the fight evenly, or in an instance where two judges scored the fight for opposing fighters and the third judge scored a draw.</p>
<p>The UFC happens to be the most popular mixed martial arts (MMA) promotion and has gone from a taboo cage-fighting organization to a world-reknown and respected sports-entertainment corporation. Like any other sport, statistical analysis and data science play a huge role in determining not only betting odds, props, and moneylines for events, but also in the UFC's matchmakers' jobs of setting up fights.</p>
<p>The aim of this tutorial is to organize and analyze UFC statistics in order to see what qualities or attributes may contribute and correlate to winning fights the most.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Import-Statements:">Import Statements:<a class="anchor-link" href="#Import-Statements:">&#182;</a></h3>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Below are all the imports utilized throughout this tutorial. The third-party libraries' documentations can be found at their respective website: <a href="https://pandas.pydata.org/pandas-docs/stable/">pandas</a>, <a href="https://docs.scipy.org/doc/numpy/user/">numpy</a>, <a href="https://matplotlib.org/contents.html">matplotlib</a>, <a href="http://scikit-learn.org/stable/documentation.html">scikit-learn</a>, and <a href="https://seaborn.pydata.org/">seaborn</a>. In addition, the popular matplotlib style <a href="https://matplotlib.org/stable/gallery/style_sheets/ggplot.html">"ggplot"</a> that mimics the <a href="https://ggplot2.tidyverse.org/">ggplot style in R</a> is used for aesthetic sugar.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="kn">from</span> <span class="nn">scipy.stats</span> <span class="kn">import</span> <span class="n">f</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">model_selection</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">linear_model</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">preprocessing</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">StandardScaler</span><span class="p">,</span> <span class="n">LabelEncoder</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span><span class="p">,</span> <span class="n">KFold</span><span class="p">,</span> <span class="n">RandomizedSearchCV</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span>
<span class="kn">from</span> <span class="nn">xgboost</span> <span class="kn">import</span> <span class="n">XGBClassifier</span> <span class="k">as</span> <span class="n">xgb</span>
<span class="kn">from</span> <span class="nn">IPython.display</span> <span class="kn">import</span> <span class="n">display</span><span class="p">,</span> <span class="n">FileLink</span>
<span class="kn">from</span> <span class="nn">datetime</span> <span class="kn">import</span> <span class="n">datetime</span>
<span class="kn">import</span> <span class="nn">random</span>
<span class="kn">import</span> <span class="nn">re</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s1">&#39;ignore&#39;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">use</span><span class="p">(</span><span class="s1">&#39;ggplot&#39;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Data-Collection-+-Parsing">Data Collection + Parsing<a class="anchor-link" href="#Data-Collection-+-Parsing">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>The data I'll be utilizing was obtained from scraping the statistics found on <a href="https://www.ufcstats.com">ufcstats.com</a>. Alternatively, pre-assembled data sets can be found across the internet, but many of these data sets aren't completely up-to-date, which is what manually scraping the data allowed me to ensure. Using scrapy and a couple custom spider scripts, I was able to get two data sets, one representing fighter stats by individual and the other representing fight stats by fight card (also referred to as event). Both of these CSV files as well as the zipped folders containing the scrapy scripts, instructions, etc. can be found below. As the instructions for scraping the data are found in the README files of the spiders, that won't be separately touched on in this tutorial.</p>
<p><em>Additional UFC stats and data that are also up-to-date but organized differently can be found in the <a href="https://github.com/KieranCanter/CMSC320FinalProject/tree/gh-pages">github repo</a> for this tutorial.</em></p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">display</span><span class="p">(</span><span class="n">FileLink</span><span class="p">(</span><span class="s1">&#39;fighters.csv&#39;</span><span class="p">,</span> <span class="n">result_html_prefix</span><span class="o">=</span><span class="s2">&quot;Fighter Data: &quot;</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span><span class="n">FileLink</span><span class="p">(</span><span class="s1">&#39;fightCards.csv&#39;</span><span class="p">,</span> <span class="n">result_html_prefix</span><span class="o">=</span><span class="s2">&quot;Fight Card Data: &quot;</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span><span class="n">FileLink</span><span class="p">(</span><span class="s1">&#39;fighterSpider.zip&#39;</span><span class="p">,</span> <span class="n">result_html_prefix</span><span class="o">=</span><span class="s2">&quot;Fighter Spider: &quot;</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span><span class="n">FileLink</span><span class="p">(</span><span class="s1">&#39;fightcardsSpider.zip&#39;</span><span class="p">,</span> <span class="n">result_html_prefix</span><span class="o">=</span><span class="s2">&quot;Fight Card Spider: &quot;</span><span class="p">))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
Fighter Data: <a href='fighters.csv' target='_blank'>fighters.csv</a><br>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
Fight Card Data: <a href='fightCards.csv' target='_blank'>fightCards.csv</a><br>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
Fighter Spider: <a href='fighterSpider.zip' target='_blank'>fighterSpider.zip</a><br>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
Fight Card Spider: <a href='fightcardsSpider.zip' target='_blank'>fightcardsSpider.zip</a><br>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Data-Management/Representation">Data Management/Representation<a class="anchor-link" href="#Data-Management/Representation">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Preprocessing-the-Data">Preprocessing the Data<a class="anchor-link" href="#Preprocessing-the-Data">&#182;</a></h3>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We first start with preprocessing the data, which includes cleaning and reorganizing the data as a pandas dataframe so that it's more readable, accessible, and so that it only contains the most relevant information we are going to utilize. Below you can see the first five rows of each dataframe along with the column headers.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;fighters.csv&#39;</span><span class="p">)</span>
<span class="n">df_cards</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;fightCards.csv&#39;</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">df_fighters</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
<span class="n">display</span><span class="p">(</span><span class="n">df_cards</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DoB</th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>name</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nov 24 1985</td>
      <td>3.72</td>
      <td>1.65</td>
      <td>5' 9"</td>
      <td>John Gunther</td>
      <td>72.0</td>
      <td>5-1-0</td>
      <td>Orthodox</td>
      <td>37%</td>
      <td>46%</td>
      <td>0.0</td>
      <td>42%</td>
      <td>7.08</td>
      <td>0%</td>
      <td>155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jul 05 1995</td>
      <td>2.80</td>
      <td>1.93</td>
      <td>6' 0"</td>
      <td>Joe Giannetti</td>
      <td>74.0</td>
      <td>6-1-0</td>
      <td>Southpaw</td>
      <td>38%</td>
      <td>40%</td>
      <td>0.0</td>
      <td>16%</td>
      <td>1.00</td>
      <td>0%</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aug 25 1974</td>
      <td>0.92</td>
      <td>0.92</td>
      <td>5' 8"</td>
      <td>Allen Berube</td>
      <td>NaN</td>
      <td>4-3-0</td>
      <td>Orthodox</td>
      <td>80%</td>
      <td>33%</td>
      <td>3.4</td>
      <td>100%</td>
      <td>6.87</td>
      <td>0%</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nov 27 1991</td>
      <td>4.49</td>
      <td>3.80</td>
      <td>5' 11"</td>
      <td>Daichi Abe</td>
      <td>71.0</td>
      <td>6-2-0</td>
      <td>Orthodox</td>
      <td>33%</td>
      <td>56%</td>
      <td>0.0</td>
      <td>50%</td>
      <td>0.33</td>
      <td>0%</td>
      <td>170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jun 26 1996</td>
      <td>6.18</td>
      <td>6.43</td>
      <td>5' 7"</td>
      <td>Diana Belbita</td>
      <td>68.0</td>
      <td>14-7-0</td>
      <td>Orthodox</td>
      <td>42%</td>
      <td>50%</td>
      <td>0.0</td>
      <td>50%</td>
      <td>0.63</td>
      <td>68%</td>
      <td>115</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_name</th>
      <th>f1</th>
      <th>f1_sig_strike_per</th>
      <th>f1_sig_strike_total</th>
      <th>f1_td_attempt</th>
      <th>f1_td_succeed</th>
      <th>f2</th>
      <th>f2_sig_strike_per</th>
      <th>f2_sig_strike_total</th>
      <th>f2_td_attempt</th>
      <th>f2_td_succeed</th>
      <th>fight_date</th>
      <th>fights_location</th>
      <th>round_format</th>
      <th>round_fought</th>
      <th>weight_class</th>
      <th>winner</th>
      <th>winning_method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Charles Rosa</td>
      <td>28%</td>
      <td>182</td>
      <td>2</td>
      <td>2</td>
      <td>Justin Jaynes</td>
      <td>47%</td>
      <td>92</td>
      <td>2</td>
      <td>2</td>
      <td>June 26 2021</td>
      <td>Las Vegas, Nevada, USA</td>
      <td>3</td>
      <td>3</td>
      <td>Featherweight</td>
      <td>Charles Rosa</td>
      <td>S-DEC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Damir Hadzovic</td>
      <td>47%</td>
      <td>219</td>
      <td>2</td>
      <td>2</td>
      <td>Yancy Medeiros</td>
      <td>51%</td>
      <td>237</td>
      <td>3</td>
      <td>2</td>
      <td>June 26 2021</td>
      <td>Las Vegas, Nevada, USA</td>
      <td>3</td>
      <td>3</td>
      <td>Lightweight</td>
      <td>Damir Hadzovic</td>
      <td>U-DEC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UFC Fight Night: Font vs. Garbrandt</td>
      <td>Damir Ismagulov</td>
      <td>47%</td>
      <td>63</td>
      <td>1</td>
      <td>0</td>
      <td>Rafael Alves</td>
      <td>44%</td>
      <td>126</td>
      <td>3</td>
      <td>2</td>
      <td>May 22 2021</td>
      <td>Las Vegas, Nevada, USA</td>
      <td>3</td>
      <td>3</td>
      <td>Lightweight</td>
      <td>Damir Ismagulov</td>
      <td>U-DEC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Julia Avila</td>
      <td>52%</td>
      <td>91</td>
      <td>4</td>
      <td>1</td>
      <td>Julija Stoliarenko</td>
      <td>42%</td>
      <td>94</td>
      <td>3</td>
      <td>1</td>
      <td>June 26 2021</td>
      <td>Las Vegas, Nevada, USA</td>
      <td>3</td>
      <td>3</td>
      <td>Women's Bantamweight</td>
      <td>Julia Avila</td>
      <td>SUB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UFC Fight Night: Hall vs. Strickland</td>
      <td>Jinh Yu Frey</td>
      <td>47%</td>
      <td>185</td>
      <td>1</td>
      <td>0</td>
      <td>Ashley Yoder</td>
      <td>38%</td>
      <td>236</td>
      <td>8</td>
      <td>0</td>
      <td>July 31 2021</td>
      <td>Las Vegas, Nevada, USA</td>
      <td>3</td>
      <td>3</td>
      <td>Women's Strawweight</td>
      <td>Jinh Yu Frey</td>
      <td>U-DEC</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Let's go over what each of these column headers mean:</p>
<table>
<thead><tr>
<th style="text-align:left">Fighter Data</th>
<th style="text-align:left"></th>
<th style="text-align:left">Fight Card Data</th>
<th style="text-align:left"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left">DoB</td>
<td style="text-align:left">Date of birth</td>
<td style="text-align:left">card_name</td>
<td style="text-align:left">Name of the card/event</td>
</tr>
<tr>
<td style="text-align:left">SApM</td>
<td style="text-align:left">Significant strikes absorbed per minute</td>
<td style="text-align:left">f1</td>
<td style="text-align:left">Fighter 1</td>
</tr>
<tr>
<td style="text-align:left">SLpM</td>
<td style="text-align:left">Significant strikes landed per minute</td>
<td style="text-align:left">f1_sig_strike_per</td>
<td style="text-align:left">Fighter 1's significant strike percentage</td>
</tr>
<tr>
<td style="text-align:left">Height</td>
<td style="text-align:left">Height of fighter</td>
<td style="text-align:left">f1_sig_strike_total</td>
<td style="text-align:left">Fighter 1's total significant strikes</td>
</tr>
<tr>
<td style="text-align:left">Name</td>
<td style="text-align:left">Name of fighter</td>
<td style="text-align:left">f1_td_attempt</td>
<td style="text-align:left"># of takedown attempts from fighter 1</td>
</tr>
<tr>
<td style="text-align:left">Reach</td>
<td style="text-align:left">Wingspan (inches)</td>
<td style="text-align:left">f1_td_succeed</td>
<td style="text-align:left"># of successful takedowns from fighter 1</td>
</tr>
<tr>
<td style="text-align:left">Record</td>
<td style="text-align:left">Professional fight record</td>
<td style="text-align:left">f2</td>
<td style="text-align:left">Fighter 2</td>
</tr>
<tr>
<td style="text-align:left">Stance</td>
<td style="text-align:left">Fighter's preferred stance</td>
<td style="text-align:left">f2_sig_strike_per</td>
<td style="text-align:left">Fighter 2's significant strike percentage</td>
</tr>
<tr>
<td style="text-align:left">strAcc</td>
<td style="text-align:left">Significant striking accuracy</td>
<td style="text-align:left">f2_sig_strike_total</td>
<td style="text-align:left">Fighter 2's total significant strikes</td>
</tr>
<tr>
<td style="text-align:left">strDef</td>
<td style="text-align:left">Significant strike defence</td>
<td style="text-align:left">f2_td_attempt</td>
<td style="text-align:left"># of takedown attempts from fighter 2</td>
</tr>
<tr>
<td style="text-align:left">subAvg</td>
<td style="text-align:left">Average submissions attempted per 15 minutes</td>
<td style="text-align:left">f2_td_succeed</td>
<td style="text-align:left"># of successful takedowns from fighter 2</td>
</tr>
<tr>
<td style="text-align:left">tdAcc</td>
<td style="text-align:left">Takedown accuracy</td>
<td style="text-align:left">fight_date</td>
<td style="text-align:left">Date of fight</td>
</tr>
<tr>
<td style="text-align:left">tdAvg</td>
<td style="text-align:left">Average takedowns landed per 15 minutes</td>
<td style="text-align:left">fights_location</td>
<td style="text-align:left">Location of fight</td>
</tr>
<tr>
<td style="text-align:left">tdDef</td>
<td style="text-align:left">Takedown defence (% of opponent TD attempts that did not succeed)</td>
<td style="text-align:left">round_format</td>
<td style="text-align:left">Max # of rounds to be fought</td>
</tr>
<tr>
<td style="text-align:left">Weight</td>
<td style="text-align:left">Most previously fought weight class</td>
<td style="text-align:left">round_fought</td>
<td style="text-align:left"># of rounds fought</td>
</tr>
<tr>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:left">weight_class</td>
<td style="text-align:left">Weight class of bout</td>
</tr>
<tr>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:left">winner</td>
<td style="text-align:left">Winning fighter</td>
</tr>
<tr>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:left">winning_method</td>
<td style="text-align:left">Method of victory</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Now let's see if there's any missing data we need to take care of.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[4]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>DoB          0
SApM         0
SLpM         0
height       0
name         0
reach     1919
record       0
stance     819
strAcc       0
strDef       0
subAvg       0
tdAcc        0
tdAvg        0
tdDef        0
weight       0
dtype: int64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As you can see, there are a ton of missing values for reach and stance. We will get to this in a moment, but let's check the other dataframe also.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_cards</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>card_name              0
f1                     0
f1_sig_strike_per      0
f1_sig_strike_total    0
f1_td_attempt          0
f1_td_succeed          0
f2                     0
f2_sig_strike_per      0
f2_sig_strike_total    0
f2_td_attempt          0
f2_td_succeed          0
fight_date             0
fights_location        0
round_format           0
round_fought           0
weight_class           0
winner                 0
winning_method         0
dtype: int64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Perfect! There aren't any missing values we have to tend to in the fight cards dataframe. Finally, let's check to see if there are any fighters with the same name that we should differentiate between.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters</span><span class="p">[</span><span class="n">df_fighters</span><span class="o">.</span><span class="n">duplicated</span><span class="p">(</span><span class="n">subset</span><span class="o">=</span><span class="s1">&#39;name&#39;</span><span class="p">,</span> <span class="n">keep</span><span class="o">=</span><span class="kc">False</span><span class="p">)]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[6]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DoB</th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>name</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281</th>
      <td>Feb 06 1965</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>5' 11"</td>
      <td>Michael McDonald</td>
      <td>NaN</td>
      <td>1-1-0</td>
      <td>Orthodox</td>
      <td>0%</td>
      <td>50%</td>
      <td>0.0</td>
      <td>0%</td>
      <td>0.00</td>
      <td>0%</td>
      <td>205</td>
    </tr>
    <tr>
      <th>615</th>
      <td>Jan 15 1991</td>
      <td>2.76</td>
      <td>2.69</td>
      <td>5' 9"</td>
      <td>Michael McDonald</td>
      <td>70.0</td>
      <td>17-4-0</td>
      <td>Orthodox</td>
      <td>42%</td>
      <td>57%</td>
      <td>1.4</td>
      <td>66%</td>
      <td>1.09</td>
      <td>52%</td>
      <td>135</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>--</td>
      <td>4.73</td>
      <td>2.00</td>
      <td>6' 1"</td>
      <td>Tony Johnson</td>
      <td>NaN</td>
      <td>11-3-0</td>
      <td>NaN</td>
      <td>53%</td>
      <td>31%</td>
      <td>0.0</td>
      <td>22%</td>
      <td>2.00</td>
      <td>0%</td>
      <td>265</td>
    </tr>
    <tr>
      <th>1389</th>
      <td>May 02 1983</td>
      <td>3.67</td>
      <td>4.00</td>
      <td>6' 2"</td>
      <td>Tony Johnson</td>
      <td>76.0</td>
      <td>7-2-0</td>
      <td>Orthodox</td>
      <td>92%</td>
      <td>22%</td>
      <td>0.0</td>
      <td>0%</td>
      <td>0.00</td>
      <td>90%</td>
      <td>205</td>
    </tr>
    <tr>
      <th>2276</th>
      <td>Oct 07 1992</td>
      <td>6.20</td>
      <td>5.83</td>
      <td>6' 0"</td>
      <td>Mike Davis</td>
      <td>72.0</td>
      <td>10-2-0</td>
      <td>Orthodox</td>
      <td>52%</td>
      <td>56%</td>
      <td>0.2</td>
      <td>53%</td>
      <td>3.04</td>
      <td>69%</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2393</th>
      <td>--</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>--</td>
      <td>Mike Davis</td>
      <td>NaN</td>
      <td>2-0-0</td>
      <td>NaN</td>
      <td>0%</td>
      <td>0%</td>
      <td>0.0</td>
      <td>0%</td>
      <td>0.00</td>
      <td>0%</td>
      <td>--</td>
    </tr>
    <tr>
      <th>2720</th>
      <td>Aug 29 1989</td>
      <td>3.33</td>
      <td>3.73</td>
      <td>5' 10"</td>
      <td>Joey Gomez</td>
      <td>71.0</td>
      <td>7-1-0</td>
      <td>Orthodox</td>
      <td>49%</td>
      <td>50%</td>
      <td>0.0</td>
      <td>28%</td>
      <td>2.00</td>
      <td>0%</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2881</th>
      <td>Jul 21 1986</td>
      <td>4.46</td>
      <td>2.44</td>
      <td>5' 10"</td>
      <td>Joey Gomez</td>
      <td>73.0</td>
      <td>6-2-0</td>
      <td>Orthodox</td>
      <td>28%</td>
      <td>55%</td>
      <td>0.0</td>
      <td>100%</td>
      <td>0.62</td>
      <td>50%</td>
      <td>135</td>
    </tr>
    <tr>
      <th>3239</th>
      <td>Mar 16 1990</td>
      <td>3.23</td>
      <td>2.98</td>
      <td>5' 4"</td>
      <td>Bruno Silva</td>
      <td>65.0</td>
      <td>12-5-2 (1 NC)</td>
      <td>Orthodox</td>
      <td>46%</td>
      <td>58%</td>
      <td>0.0</td>
      <td>31%</td>
      <td>2.89</td>
      <td>64%</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3364</th>
      <td>Jul 13 1989</td>
      <td>4.58</td>
      <td>4.31</td>
      <td>6' 0"</td>
      <td>Bruno Silva</td>
      <td>74.0</td>
      <td>22-8-0</td>
      <td>Orthodox</td>
      <td>48%</td>
      <td>44%</td>
      <td>0.0</td>
      <td>18%</td>
      <td>0.66</td>
      <td>71%</td>
      <td>185</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As you can see, there are five different names across these ten fighters. Let's fix this issue first. We'll add the weight class to one of every two of these fighters to differentiate them and avoid duplicates. Since the Mike Davis entry with a record of 2-0-0 is missing all of the associated data, we'll just drop him.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">446</span><span class="p">,</span> <span class="mi">4</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Michael McDonald 135&quot;</span>
<span class="n">df_fighters</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">1318</span><span class="p">,</span> <span class="mi">4</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Tony Johnson 265&quot;</span>
<span class="n">df_fighters</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">2092</span><span class="p">,</span> <span class="mi">4</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Joey Gomez 155&quot;</span>
<span class="n">df_fighters</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">3300</span><span class="p">,</span> <span class="mi">4</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Bruno Silva 185&quot;</span>
<span class="n">df_fighters</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="mi">2404</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>To make things arithmetically easier, we'll convert all the percentages (string objects) to decimal values.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">per2dec</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">columns</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">columns</span><span class="p">:</span>
        <span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">strip</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
        <span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_numeric</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">])</span> <span class="o">/</span> <span class="mi">100</span>

<span class="n">per2dec</span><span class="p">(</span><span class="n">df_fighters</span><span class="p">,</span> <span class="p">[</span><span class="s1">&#39;strAcc&#39;</span><span class="p">,</span> <span class="s1">&#39;strDef&#39;</span><span class="p">,</span> <span class="s1">&#39;tdAcc&#39;</span><span class="p">,</span> <span class="s1">&#39;tdDef&#39;</span><span class="p">])</span>
<span class="n">df_fighters</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[8]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DoB</th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>name</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nov 24 1985</td>
      <td>3.72</td>
      <td>1.65</td>
      <td>5' 9"</td>
      <td>John Gunther</td>
      <td>72.0</td>
      <td>5-1-0</td>
      <td>Orthodox</td>
      <td>0.37</td>
      <td>0.46</td>
      <td>0.0</td>
      <td>0.42</td>
      <td>7.08</td>
      <td>0.00</td>
      <td>155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jul 05 1995</td>
      <td>2.80</td>
      <td>1.93</td>
      <td>6' 0"</td>
      <td>Joe Giannetti</td>
      <td>74.0</td>
      <td>6-1-0</td>
      <td>Southpaw</td>
      <td>0.38</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aug 25 1974</td>
      <td>0.92</td>
      <td>0.92</td>
      <td>5' 8"</td>
      <td>Allen Berube</td>
      <td>NaN</td>
      <td>4-3-0</td>
      <td>Orthodox</td>
      <td>0.80</td>
      <td>0.33</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>6.87</td>
      <td>0.00</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nov 27 1991</td>
      <td>4.49</td>
      <td>3.80</td>
      <td>5' 11"</td>
      <td>Daichi Abe</td>
      <td>71.0</td>
      <td>6-2-0</td>
      <td>Orthodox</td>
      <td>0.33</td>
      <td>0.56</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jun 26 1996</td>
      <td>6.18</td>
      <td>6.43</td>
      <td>5' 7"</td>
      <td>Diana Belbita</td>
      <td>68.0</td>
      <td>14-7-0</td>
      <td>Orthodox</td>
      <td>0.42</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.63</td>
      <td>0.68</td>
      <td>115</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>All of our fighter stat percentages are now decimals! Next, for all the fighters missing a substantial amount of data, which we will define as having 0 strDef, tdAvg, tdAcc, tdDef, and subAvg, we will simply remove them.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[9]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span> <span class="o">=</span> <span class="n">df_fighters</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="o">~</span><span class="p">(</span>
    <span class="p">(</span><span class="n">df_fighters</span><span class="p">[</span><span class="s2">&quot;strDef&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span>
    <span class="p">(</span><span class="n">df_fighters</span><span class="p">[</span><span class="s2">&quot;tdAvg&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span>
    <span class="p">(</span><span class="n">df_fighters</span><span class="p">[</span><span class="s2">&quot;tdAcc&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span>
    <span class="p">(</span><span class="n">df_fighters</span><span class="p">[</span><span class="s2">&quot;tdDef&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span>
    <span class="p">(</span><span class="n">df_fighters</span><span class="p">[</span><span class="s2">&quot;subAvg&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">))]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Something to be mindful of is noise in your data. Often times, fighters with no DoB in their statistics page means they've only fought one match in the UFC and are no longer with the organization and/or they fought in the very early days of the promotion. The sport has changed drastically since the 90's/early 2000's, so keeping these fighters in the dataframe may just cause more noise than it would benefit us. While we're at it, we'll strip the birth date entries to include just the year the fighter was born since the exact month and day aren't significant.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="o">~</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;DoB&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;--&#39;</span><span class="p">)]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

<span class="k">def</span> <span class="nf">get_birth_year</span><span class="p">(</span><span class="n">dob</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">datetime</span><span class="o">.</span><span class="n">strptime</span><span class="p">(</span><span class="n">dob</span><span class="p">,</span> <span class="s1">&#39;%b </span><span class="si">%d</span><span class="s1"> %Y&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">year</span>

<span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;birth_year&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;DoB&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">get_birth_year</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;DoB&#39;</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Another minor tweak we'll make is setting the index of the dataframe as the name column.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">set_index</span><span class="p">(</span><span class="s1">&#39;name&#39;</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[11]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
      <th>birth_year</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>John Gunther</th>
      <td>3.72</td>
      <td>1.65</td>
      <td>5' 9"</td>
      <td>72.0</td>
      <td>5-1-0</td>
      <td>Orthodox</td>
      <td>0.37</td>
      <td>0.46</td>
      <td>0.0</td>
      <td>0.42</td>
      <td>7.08</td>
      <td>0.00</td>
      <td>155</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>Joe Giannetti</th>
      <td>2.80</td>
      <td>1.93</td>
      <td>6' 0"</td>
      <td>74.0</td>
      <td>6-1-0</td>
      <td>Southpaw</td>
      <td>0.38</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>155</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>Allen Berube</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>5' 8"</td>
      <td>NaN</td>
      <td>4-3-0</td>
      <td>Orthodox</td>
      <td>0.80</td>
      <td>0.33</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>6.87</td>
      <td>0.00</td>
      <td>155</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>Daichi Abe</th>
      <td>4.49</td>
      <td>3.80</td>
      <td>5' 11"</td>
      <td>71.0</td>
      <td>6-2-0</td>
      <td>Orthodox</td>
      <td>0.33</td>
      <td>0.56</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>170</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>Diana Belbita</th>
      <td>6.18</td>
      <td>6.43</td>
      <td>5' 7"</td>
      <td>68.0</td>
      <td>14-7-0</td>
      <td>Orthodox</td>
      <td>0.42</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.63</td>
      <td>0.68</td>
      <td>115</td>
      <td>1996</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Now finally we'll revisit the problem of missing values we had before. Reach is a pretty crucial physical trait that can have a big effect on the outcome of a fight. We need to figure out a way to deal with these missing values.</p>
<p>As with other physical characteristics like foot size, hand size, inseam, etc., wingspan has a lot to do with the height of the individual. While not perfect, we can estimate a reach value that's as statistically probable as possible by calculating the median reaches of every height (5'0", 5'1", ...) and attributing that median to the reach value of each fighter that's missing said value.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">median_wingspans</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;height&#39;</span><span class="p">)[</span><span class="s1">&#39;reach&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span>
<span class="n">display</span><span class="p">(</span><span class="n">median_wingspans</span><span class="p">)</span>

<span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;reach&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;reach&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;height&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="n">median_wingspans</span><span class="p">))</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;There are still </span><span class="si">{}</span><span class="s2"> missing reach values&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;reach&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedText jp-OutputArea-output " data-mime-type="text/plain">
<pre>height
--        70.0
5&#39; 0&#34;     61.5
5&#39; 1&#34;     62.0
5&#39; 10&#34;    72.0
5&#39; 11&#34;    73.0
5&#39; 2&#34;     63.0
5&#39; 3&#34;     64.0
5&#39; 4&#34;     65.0
5&#39; 5&#34;     66.0
5&#39; 6&#34;     67.0
5&#39; 7&#34;     69.0
5&#39; 8&#34;     70.0
5&#39; 9&#34;     71.0
6&#39; 0&#34;     74.0
6&#39; 1&#34;     75.0
6&#39; 10&#34;     NaN
6&#39; 11&#34;    84.0
6&#39; 2&#34;     75.0
6&#39; 3&#34;     77.0
6&#39; 4&#34;     78.0
6&#39; 5&#34;     79.0
6&#39; 6&#34;     79.0
6&#39; 7&#34;     80.0
6&#39; 8&#34;     80.0
7&#39; 2&#34;      NaN
7&#39; 5&#34;      NaN
Name: reach, dtype: float64</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>There are still 5 missing reach values
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As seen from the output above, we still have 5 missing reach values. Compared to the 1919 that we started with, that's a substantial decrease. Since 5 is so insignificant to the gross total number of entries we have, we can just drop these entries.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[13]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">subset</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;reach&#39;</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As for the missing stance values, these aren't as weighted in determining the outcome of a fight and are normally just fighter preference. We'll replace these by just getting a percentage of each stance out of the total</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">display</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;stance&#39;</span><span class="p">)[</span><span class="s1">&#39;stance&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">())</span>
<span class="n">stance_total</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;stance&#39;</span><span class="p">)[</span><span class="s1">&#39;stance&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;There are </span><span class="si">{}</span><span class="s2"> total stance entries&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">stance_total</span><span class="p">))</span>

<span class="n">display</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;stance&#39;</span><span class="p">)[</span><span class="s1">&#39;stance&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span> <span class="o">/</span> <span class="n">stance_total</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedText jp-OutputArea-output " data-mime-type="text/plain">
<pre>stance
Open Stance       5
Orthodox       2060
Sideways          1
Southpaw        477
Switch          154
Name: stance, dtype: int64</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>There are 2697 total stance entries
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedText jp-OutputArea-output " data-mime-type="text/plain">
<pre>stance
Open Stance    0.001854
Orthodox       0.763812
Sideways       0.000371
Southpaw       0.176863
Switch         0.057100
Name: stance, dtype: float64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We've calculated the decimal values that represent the percentage of each stance's frequency relative to the total. Using those as weights, we'll replace the missing stance values with one of the existing stances based on those decimal percentages.</p>
<p>As a demo, you can see a list of 50 random choices. Unsurprisingly, orthodox comes up the most frequently, with some southpaw, swithc, and the occasional open stance if RNG permits.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">stance_list</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Open Stance&quot;</span><span class="p">,</span> <span class="s2">&quot;Orthodox&quot;</span><span class="p">,</span> <span class="s2">&quot;Sideways&quot;</span><span class="p">,</span> <span class="s2">&quot;Southpaw&quot;</span><span class="p">,</span> <span class="s2">&quot;Switch&quot;</span><span class="p">]</span>
<span class="n">weight_list</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.001854</span><span class="p">,</span> <span class="mf">0.763812</span><span class="p">,</span> <span class="mf">0.000371</span><span class="p">,</span> <span class="mf">0.176863</span><span class="p">,</span> <span class="mf">0.057100</span><span class="p">]</span>

<span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;stance&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">random</span><span class="o">.</span><span class="n">choices</span><span class="p">(</span><span class="n">stance_list</span><span class="p">,</span> <span class="n">weights</span><span class="o">=</span><span class="n">weight_list</span><span class="p">,</span> <span class="n">k</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)[</span><span class="mi">0</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">random</span><span class="o">.</span><span class="n">choices</span><span class="p">(</span><span class="n">stance_list</span><span class="p">,</span> <span class="n">weights</span><span class="o">=</span><span class="n">weight_list</span><span class="p">,</span> <span class="n">k</span> <span class="o">=</span> <span class="mi">50</span><span class="p">))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>[&#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Switch&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Switch&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Southpaw&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Southpaw&#39;, &#39;Southpaw&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;, &#39;Orthodox&#39;]
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Wonderful! We have no more missing values, right? Let's check to be sure.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[16]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">display</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="n">df_fighters_clean</span><span class="p">[</span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;height&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;--&#39;</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedText jp-OutputArea-output " data-mime-type="text/plain">
<pre>SApM          0
SLpM          0
height        0
reach         0
record        0
stance        0
strAcc        0
strDef        0
subAvg        0
tdAcc         0
tdAvg         0
tdDef         0
weight        0
birth_year    0
dtype: int64</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[16]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
      <th>birth_year</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Amador Ramirez</th>
      <td>2.07</td>
      <td>4.93</td>
      <td>--</td>
      <td>70.0</td>
      <td>5-4-0</td>
      <td>Orthodox</td>
      <td>0.51</td>
      <td>0.69</td>
      <td>0.0</td>
      <td>0.33</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>135</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>Matt Ricehouse</th>
      <td>4.80</td>
      <td>3.70</td>
      <td>--</td>
      <td>70.0</td>
      <td>6-1-0</td>
      <td>Orthodox</td>
      <td>0.44</td>
      <td>0.47</td>
      <td>0.0</td>
      <td>0.22</td>
      <td>1.00</td>
      <td>0.81</td>
      <td>155</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>Logan Nail</th>
      <td>2.27</td>
      <td>1.93</td>
      <td>--</td>
      <td>70.0</td>
      <td>1-1-0</td>
      <td>Orthodox</td>
      <td>0.51</td>
      <td>0.39</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.37</td>
      <td>185</td>
      <td>1989</td>
    </tr>
    <tr>
      <th>Lee Higgins</th>
      <td>3.68</td>
      <td>1.02</td>
      <td>--</td>
      <td>70.0</td>
      <td>2-1-0</td>
      <td>Orthodox</td>
      <td>0.26</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>155</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>Hiroshi Izumi</th>
      <td>2.65</td>
      <td>1.95</td>
      <td>--</td>
      <td>70.0</td>
      <td>4-2-0</td>
      <td>Orthodox</td>
      <td>0.37</td>
      <td>0.66</td>
      <td>0.5</td>
      <td>0.70</td>
      <td>3.35</td>
      <td>1.00</td>
      <td>205</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>Neal Ewing</th>
      <td>1.93</td>
      <td>2.27</td>
      <td>--</td>
      <td>70.0</td>
      <td>6-0-0</td>
      <td>Orthodox</td>
      <td>0.60</td>
      <td>0.48</td>
      <td>0.0</td>
      <td>0.62</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>185</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>TJ Cook</th>
      <td>3.18</td>
      <td>2.30</td>
      <td>--</td>
      <td>70.0</td>
      <td>13-5-0</td>
      <td>Orthodox</td>
      <td>0.47</td>
      <td>0.54</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>205</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>Joe Duarte</th>
      <td>4.00</td>
      <td>2.27</td>
      <td>--</td>
      <td>70.0</td>
      <td>10-4-0</td>
      <td>Orthodox</td>
      <td>0.38</td>
      <td>0.53</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>3.00</td>
      <td>0.69</td>
      <td>155</td>
      <td>1977</td>
    </tr>
    <tr>
      <th>Billy Goff</th>
      <td>4.15</td>
      <td>9.95</td>
      <td>--</td>
      <td>70.0</td>
      <td>8-2-0</td>
      <td>Switch</td>
      <td>0.45</td>
      <td>0.59</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>8.29</td>
      <td>1.00</td>
      <td>170</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>Edward Faaloloto</th>
      <td>6.25</td>
      <td>2.28</td>
      <td>--</td>
      <td>70.0</td>
      <td>2-5-0</td>
      <td>Orthodox</td>
      <td>0.32</td>
      <td>0.44</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>1.01</td>
      <td>0.33</td>
      <td>155</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>Bryan Travers</th>
      <td>3.93</td>
      <td>2.33</td>
      <td>--</td>
      <td>70.0</td>
      <td>15-4-0</td>
      <td>Orthodox</td>
      <td>0.48</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.28</td>
      <td>2.00</td>
      <td>0.63</td>
      <td>155</td>
      <td>1983</td>
    </tr>
    <tr>
      <th>Maka Watson</th>
      <td>1.60</td>
      <td>0.93</td>
      <td>--</td>
      <td>70.0</td>
      <td>4-2-0</td>
      <td>Orthodox</td>
      <td>0.37</td>
      <td>0.22</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>0.33</td>
      <td>155</td>
      <td>1984</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>It turns out we still have some missing height data that wasn't caught before because it's replaced with "--" strings. After some research, most of these fighters can be found to have a height of 5'7", so we'll simply use that number to replace the few missing values we have. We'll then convert height from feet and inches to centimeters since that's a more convenient metric to mathematically work with.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[17]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;height&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">({</span><span class="s2">&quot;--&quot;</span><span class="p">:</span> <span class="s2">&quot;5</span><span class="se">\&#39;</span><span class="s2"> 7</span><span class="se">\&quot;</span><span class="s2">&quot;</span><span class="p">},</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># Method to convert feet&#39;inches&quot; to cm</span>
<span class="k">def</span> <span class="nf">convert_to_cm</span><span class="p">(</span><span class="n">height</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">height</span> <span class="ow">is</span> <span class="n">np</span><span class="o">.</span><span class="n">NaN</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">height</span>
    <span class="k">elif</span> <span class="nb">len</span><span class="p">(</span><span class="n">height</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;&#39;&quot;</span><span class="p">))</span> <span class="o">==</span> <span class="mi">2</span><span class="p">:</span>
        <span class="n">feet</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="n">height</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;&#39;&quot;</span><span class="p">)[</span><span class="mi">0</span><span class="p">])</span>
        <span class="n">inches</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">height</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;&#39;&quot;</span><span class="p">)[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span> <span class="s1">&#39;&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;&quot;&#39;</span><span class="p">,</span><span class="s1">&#39;&#39;</span><span class="p">))</span>
        <span class="k">return</span> <span class="p">(</span><span class="n">feet</span> <span class="o">*</span> <span class="mf">30.48</span><span class="p">)</span> <span class="o">+</span> <span class="p">(</span><span class="n">inches</span> <span class="o">*</span> <span class="mf">2.54</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">height</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;&quot;&#39;</span><span class="p">,</span><span class="s1">&#39;&#39;</span><span class="p">))</span> <span class="o">*</span> <span class="mf">2.54</span>

<span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;height&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;height&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">convert_to_cm</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Since we had disguised missing values in the height column, let's check to see if there are any in the weight column.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[18]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;weight&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;--&#39;</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[18]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>reach</th>
      <th>record</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
      <th>birth_year</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Great, the dataframe is empty, which means there are no missing weight values.</p>
<p>Next we'll split up the record so it's more integer friendly. We'll have to specify a new method to split the no contest outcomes because they are encaptured in parentheses, unlike the win-loss-draw numbers. After creating the appropriate columns, we'll drop the defunct record column.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[19]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;record&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;record&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; \(&#39;</span><span class="p">,</span> <span class="s1">&#39;-(&#39;</span><span class="p">,</span> <span class="n">regex</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_fighters_clean</span><span class="p">[[</span><span class="s1">&#39;win&#39;</span><span class="p">,</span> <span class="s1">&#39;lose&#39;</span><span class="p">,</span> <span class="s1">&#39;draw&#39;</span><span class="p">,</span> <span class="s1">&#39;nc&#39;</span><span class="p">]]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;record&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;-&#39;</span><span class="p">,</span> <span class="n">expand</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">split_nc</span><span class="p">(</span><span class="n">nc</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">re</span><span class="o">.</span><span class="n">findall</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;\d+&quot;</span><span class="p">,</span> <span class="n">nc</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">IGNORECASE</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
    
<span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;nc&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="p">[</span><span class="s1">&#39;nc&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">split_nc</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">if</span> <span class="n">x</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="k">else</span> <span class="mi">0</span><span class="p">)</span>
<span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;record&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[19]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SApM</th>
      <th>SLpM</th>
      <th>height</th>
      <th>reach</th>
      <th>stance</th>
      <th>strAcc</th>
      <th>strDef</th>
      <th>subAvg</th>
      <th>tdAcc</th>
      <th>tdAvg</th>
      <th>tdDef</th>
      <th>weight</th>
      <th>birth_year</th>
      <th>win</th>
      <th>lose</th>
      <th>draw</th>
      <th>nc</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>John Gunther</th>
      <td>3.72</td>
      <td>1.65</td>
      <td>175.26</td>
      <td>72.0</td>
      <td>Orthodox</td>
      <td>0.37</td>
      <td>0.46</td>
      <td>0.0</td>
      <td>0.42</td>
      <td>7.08</td>
      <td>0.00</td>
      <td>155</td>
      <td>1985</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Joe Giannetti</th>
      <td>2.80</td>
      <td>1.93</td>
      <td>182.88</td>
      <td>74.0</td>
      <td>Southpaw</td>
      <td>0.38</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>155</td>
      <td>1995</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Allen Berube</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>172.72</td>
      <td>70.0</td>
      <td>Orthodox</td>
      <td>0.80</td>
      <td>0.33</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>6.87</td>
      <td>0.00</td>
      <td>155</td>
      <td>1974</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Daichi Abe</th>
      <td>4.49</td>
      <td>3.80</td>
      <td>180.34</td>
      <td>71.0</td>
      <td>Orthodox</td>
      <td>0.33</td>
      <td>0.56</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>170</td>
      <td>1991</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Diana Belbita</th>
      <td>6.18</td>
      <td>6.43</td>
      <td>170.18</td>
      <td>68.0</td>
      <td>Orthodox</td>
      <td>0.42</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.63</td>
      <td>0.68</td>
      <td>115</td>
      <td>1996</td>
      <td>14</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Finally, let's make sure our columns are all of the proper types.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[20]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[20]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>SApM          float64
SLpM          float64
height        float64
reach         float64
stance         object
strAcc        float64
strDef        float64
subAvg        float64
tdAcc         float64
tdAvg         float64
tdDef         float64
weight         object
birth_year      int64
win            object
lose           object
draw           object
nc             object
dtype: object</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>It seems we have a few columns that are string objects when they should be integers. To amend this, we'll create a simple function that converts strings to int for every appropriate column in the dataframe. The only one that should remain as a string is the stance column. The properly converted data types can be seen below.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[21]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">str2int</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">columns</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">columns</span><span class="p">:</span>
        <span class="n">df</span><span class="p">[</span><span class="n">column</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
        
        
<span class="n">str2int</span><span class="p">(</span><span class="n">df_fighters_clean</span><span class="p">,</span> <span class="p">[</span><span class="s1">&#39;win&#39;</span><span class="p">,</span> <span class="s1">&#39;lose&#39;</span><span class="p">,</span> <span class="s1">&#39;draw&#39;</span><span class="p">,</span> <span class="s1">&#39;nc&#39;</span><span class="p">,</span> <span class="s1">&#39;weight&#39;</span><span class="p">])</span>

<span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[21]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>SApM          float64
SLpM          float64
height        float64
reach         float64
stance         object
strAcc        float64
strDef        float64
subAvg        float64
tdAcc         float64
tdAvg         float64
tdDef         float64
weight          int32
birth_year      int64
win             int32
lose            int32
draw            int32
nc              int32
dtype: object</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Now that our fighter data is completely cleaned up, we can move on to the fight card dataframe. Luckily, we've already written functions that perform most of the cleaning we'll do on the fight card data. We'll start by reusing our per2dec function from before to convert the two significant strike percentage columns to decimal values.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[22]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">per2dec</span><span class="p">(</span><span class="n">df_cards</span><span class="p">,</span> <span class="p">[</span><span class="s1">&#39;f1_sig_strike_per&#39;</span><span class="p">,</span> <span class="s1">&#39;f2_sig_strike_per&#39;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We can again reuse the get_birth_year function's structure to get just the year of each fight.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[23]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">get_fight_year</span><span class="p">(</span><span class="n">dof</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">datetime</span><span class="o">.</span><span class="n">strptime</span><span class="p">(</span><span class="n">dof</span><span class="p">,</span> <span class="s1">&#39;%B </span><span class="si">%d</span><span class="s1"> %Y&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">year</span>

<span class="n">df_cards</span><span class="p">[</span><span class="s1">&#39;fight_year&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_cards</span><span class="p">[</span><span class="s1">&#39;fight_date&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">get_fight_year</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">df_cards</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;fight_date&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Another tricky characteristic we have to be weary of here is the fact that UFCStats always attributes the winner to "Fighter 1" (f1). To fix this, we'll just randomly swap f1 and f2 for half of the dataset so that about 50% of winners belong to each f1 and f2. We'll check to make sure they were rearranged properly below.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[24]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">swap_indices</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">choice</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">df_cards</span><span class="p">),</span> <span class="n">size</span><span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">df_cards</span><span class="p">)</span> <span class="o">//</span> <span class="mi">2</span><span class="p">,</span> <span class="n">replace</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
<span class="n">df_cards</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">swap_indices</span><span class="p">,</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">]]</span> <span class="o">=</span> <span class="n">df_cards</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">swap_indices</span><span class="p">,</span> <span class="p">[</span><span class="mi">6</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>

<span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;winner&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;winner&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;f1&quot;</span><span class="p">]</span>
<span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;winner&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;winner&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
<span class="n">df_cards</span><span class="p">[</span><span class="s2">&quot;winner&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[24]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>1    3398
0    3397
Name: winner, dtype: int64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Since we had to change some names of fighters earlier due to duplication, we'll repeat the same process here.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[25]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_cards_clean</span> <span class="o">=</span> <span class="n">df_cards</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">&#39;f1&#39;</span><span class="p">,</span> <span class="s1">&#39;f2&#39;</span><span class="p">]:</span>
    <span class="n">df_cards_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Michael McDonald&#39;</span><span class="p">)</span> <span class="o">&amp;</span> 
                    <span class="p">(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Bantamweight&#39;</span><span class="p">),</span> <span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Michael McDonald 135&quot;</span>
    
    <span class="n">df_cards_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Tony Johnson&#39;</span><span class="p">)</span> <span class="o">&amp;</span> 
                    <span class="p">(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Heavyweight&#39;</span><span class="p">),</span> <span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Tony Johnson 265&quot;</span>
    
    <span class="n">df_cards_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Joey Gomez&#39;</span><span class="p">)</span> <span class="o">&amp;</span> 
                    <span class="p">(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Welterweight&#39;</span><span class="p">),</span> <span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Joey Gomez 155&quot;</span>
    
    <span class="n">df_cards_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Bruno Silva&#39;</span><span class="p">)</span> <span class="o">&amp;</span> 
                    <span class="p">(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;Light Heavyweight&#39;</span><span class="p">),</span> <span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Bruno Silva 185&quot;</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Next we'll compile a list of all the fighters from the df_fighters_clean dataframe. As a limitation of this DF, we'll drop the fights that don't have the fighters from that dataframe.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[26]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">all_fighters</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">index</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>

<span class="n">df_cards_clean</span> <span class="o">=</span> <span class="n">df_cards_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s2">&quot;f1&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">all_fighters</span><span class="p">))</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s2">&quot;f2&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">all_fighters</span><span class="p">))]</span>
<span class="n">df_cards_clean</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;We had </span><span class="si">{}</span><span class="s2"> cards initially. After clean up we have </span><span class="si">{}</span><span class="s2"> cards&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">df_cards</span><span class="p">),</span> <span class="nb">len</span><span class="p">(</span><span class="n">df_cards_clean</span><span class="p">)))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>We had 6795 cards initially. After clean up we have 6590 cards
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We'll create two new dataframes now to get the stats of fighter 1 and fighter 2 separately from the df_fighters_clean dataframe. We'll then rejoin these dataframes and concatenate it with the df_cards_clean dataframe to get a single, final dataframe we can examine.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[27]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Split</span>
<span class="n">df_f1</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;f1&#39;</span><span class="p">]]</span>
<span class="n">df_f1</span> <span class="o">=</span> <span class="n">df_f1</span><span class="o">.</span><span class="n">add_suffix</span><span class="p">(</span><span class="s1">&#39;_f1&#39;</span><span class="p">)</span>
<span class="n">df_f2</span> <span class="o">=</span> <span class="n">df_fighters_clean</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_cards_clean</span><span class="p">[</span><span class="s1">&#39;f2&#39;</span><span class="p">]]</span>
<span class="n">df_f2</span> <span class="o">=</span> <span class="n">df_f2</span><span class="o">.</span><span class="n">add_suffix</span><span class="p">(</span><span class="s1">&#39;_f2&#39;</span><span class="p">)</span>

<span class="c1"># Join</span>
<span class="n">df_f1</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_f2</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_final</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">df_cards_clean</span><span class="p">,</span> <span class="n">df_f1</span><span class="p">,</span> <span class="n">df_f2</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">sort</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="c1"># Rename columns</span>
<span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;f1_age_when_fight&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;fight_year&#39;</span><span class="p">]</span> <span class="o">-</span> <span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;birth_year_f1&#39;</span><span class="p">]</span>
<span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;f2_age_when_fight&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;fight_year&#39;</span><span class="p">]</span> <span class="o">-</span> <span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;birth_year_f2&#39;</span><span class="p">]</span>

<span class="n">df_final</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[27]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_name</th>
      <th>f1</th>
      <th>f1_sig_strike_per</th>
      <th>f1_sig_strike_total</th>
      <th>f1_td_attempt</th>
      <th>f1_td_succeed</th>
      <th>f2</th>
      <th>f2_sig_strike_per</th>
      <th>f2_sig_strike_total</th>
      <th>f2_td_attempt</th>
      <th>...</th>
      <th>tdAvg_f2</th>
      <th>tdDef_f2</th>
      <th>weight_f2</th>
      <th>birth_year_f2</th>
      <th>win_f2</th>
      <th>lose_f2</th>
      <th>draw_f2</th>
      <th>nc_f2</th>
      <th>f1_age_when_fight</th>
      <th>f2_age_when_fight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Justin Jaynes</td>
      <td>0.28</td>
      <td>182.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Charles Rosa</td>
      <td>0.47</td>
      <td>92.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.67</td>
      <td>0.38</td>
      <td>145</td>
      <td>1986</td>
      <td>14</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>32.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Damir Hadzovic</td>
      <td>0.47</td>
      <td>219.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Yancy Medeiros</td>
      <td>0.51</td>
      <td>237.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.33</td>
      <td>0.73</td>
      <td>155</td>
      <td>1987</td>
      <td>15</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>35.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UFC Fight Night: Font vs. Garbrandt</td>
      <td>Damir Ismagulov</td>
      <td>0.47</td>
      <td>63.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Rafael Alves</td>
      <td>0.44</td>
      <td>126.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.41</td>
      <td>0.60</td>
      <td>155</td>
      <td>1990</td>
      <td>20</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>30.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UFC Fight Night: Gane vs. Volkov</td>
      <td>Julija Stoliarenko</td>
      <td>0.52</td>
      <td>91.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>Julia Avila</td>
      <td>0.42</td>
      <td>94.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.67</td>
      <td>0.61</td>
      <td>135</td>
      <td>1988</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>28.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UFC Fight Night: Hall vs. Strickland</td>
      <td>Ashley Yoder</td>
      <td>0.47</td>
      <td>185.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Jinh Yu Frey</td>
      <td>0.38</td>
      <td>236.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0.61</td>
      <td>0.88</td>
      <td>115</td>
      <td>1985</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>34.0</td>
      <td>36.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>For record's sake, we'll output this final dataframe to a CSV.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[28]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_final</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;cleaned_ufc_stats.csv&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="n">display</span><span class="p">(</span><span class="n">FileLink</span><span class="p">(</span><span class="s1">&#39;cleaned_ufc_stats.csv&#39;</span><span class="p">,</span> <span class="n">result_html_prefix</span><span class="o">=</span><span class="s2">&quot;Cleaned UFC Stats: &quot;</span><span class="p">))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
Cleaned UFC Stats: <a href='cleaned_ufc_stats.csv' target='_blank'>cleaned_ufc_stats.csv</a><br>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Exploratory-Data-Analysis">Exploratory Data Analysis<a class="anchor-link" href="#Exploratory-Data-Analysis">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Now that our data is all cleaned up, we can finally analyze it. Let's start by looking at statistics surrounding the age of fighters. The first set of histograms displays the distribution of fighters ages throughout the UFC. The second set of histograms displays number of wins versus the age of the fighter, in decreasing order.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[29]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">histplot</span><span class="p">(</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;f1_age_when_fight&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">kde</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 1 Ages&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">histplot</span><span class="p">(</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;f2_age_when_fight&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">kde</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 2 Ages&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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fPOb38QPfvCDilwrUb2ptjbs6NGjuOqqq7BixQr84he/gNVqnfYxbMNqC3vOaFq33347Nm/ejMsuuwwf/ehHsXfvXjz11FMAirsbzVu4cCFEUcSvf/1rbN68GVarFR6PB/fddx9uvvlmeL1ebNy4EWazGYcPH8Z//ud/4oc//OGM4+3u7kY0GsXJkycBAPv37weAcxqWvEAggJ/97Gd4/PHHcdFFF405tnjxYtx555346U9/is985jO4/fbbcffdd+P888/HZZddhv/4j//A7373uzE/h29+85u49tprcdttt+Gzn/0sXC4Xjhw5gp/+9Kf43ve+B0VRcMcdd+DP/uzPsHjxYoRCIfzmN78pDBsQUWlVUxt26NAhbNiwARdffDG2b9+OoaGhwrGmpqYxPXt5bMNqkN6T3kgfM12G/tBDD2ltbW2aLMvatddeq/3whz/UABSWVk80+b6np0cDoD3zzDOF733729/W2traNFEUxyxD37Fjh3bFFVdoNptNc7lc2iWXXKL93d/9XeH4+vXrtZtvvrmoa1u/fv05S97HxzH+2qxW66TL1D/xiU9o73//+zVNyy1Dv/POO7XGxsbCMvT77rtPczqdYx6zZ88e7ZprrtGcTmdhmfmXv/xlLZPJaIlEQvvkJz+pLVq0SLNarVpTU5N2/fXXaydPnizq+oiodtuwu+++e8L2CxMsRBh9bWzDaougaSOD0EQz8M1vfhOPPPLImLu6enXTTTfhwIEDePnll/UOhYiKxDbsXWzDjIfDmjStTCaDBx98EB/72MfgcDjwzDPP4IEHHsAtt9yid2gV19vbix07duDqq6+GJEn45S9/iaeeegrf+9739A6NiCbBNuxdbMOqA3vOaFrZbBbXXXcdXn75ZQwPD2Px4sX4zGc+g6997Wt1t1Hw2bNnsXnzZhw8eBDJZBJLly7Fl770JXzuc5/TOzQimgTbsHexDasOTM6IiIiIDISlNIiIiIgMhMkZERERkYEwOSMiIiIykJqaCdnb21uW5/X7/RgcHCzLcxsJr7N21MM1AkBbW5veIZRUudqwPKO8LxiH8eIwQgz1FsdU7Rd7zoiIiIgMhMkZERERkYEwOSMiIiIyECZnRERERAbC5IyIiIjIQJicERERERkIkzMiIiIiA2FyRkRERGQgTM6IiIiIDITJGREREZGBMDkjIiIiMhAm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YXJGREREZCAmvQMgGk0QhAn/n4iIqF4wOSPDEAQB+3qiCCczAIA2twyACRoREdUXJmdkKOFkBoF4Ljlzy3x7EhFR/eGcMyIiIiIDYXJGREREZCBMzoiIiIgMhMkZERERkYFwxjUREVEVE4RzSw9pmqZTNFQKTM6IiIiqmEc248XTMYTiaQCA127GmjYHE7QqxuSMiIioyoXiGQzE0nqHQSXCOWdEREREBsLkjIiIiMhAmJwRERERGQiTMyIiIiIDYXJGREREZCBMzoiIiIgMhKU0iIiIwEKuZBxMzoiIqO4JgoCu3hhC8QwAFnIlfTE5IyIiAgu5knFwzhkRERGRgTA5IyIiIjIQJmdEREREBsLkjIiIiMhAmJwRERERGQiTMyIiIiIDYXJGREREZCBMzoiIiIgMhMkZERERkYEwOSMiIiIyEG7fRDRDozdH5r57RERUakzOiGZAEATs64kinMzAI5uxtsPJBI2IiEqKyRnRDIWTGQTiGb3DICIq2ugef4C9/kbH5IyIiKiGCYKArt4YQiM3lV67GWvaHEzQDIzJGRERUY0LxTMYiKX1DoOKxNWaRERERAbC5IyIiIjIQJicERERERkI55wRERVBVVXceeedaGhowJ133oloNIpt27ZhYGAATU1NuO222+B0OgEAO3bswO7duyGKIm688UasWrVK3+CJqKqw54yIqAi//vWv0d7eXvj3zp07sXLlSmzfvh0rV67Ezp07AQCnTp3C3r178dBDD+HrX/86nnjiCaiqqlPURFSNmJwREU1jaGgIr7zyCq655prC97q6urB+/XoAwPr169HV1VX4/rp162A2m9Hc3IyWlhZ0d3frEjcRVScmZ0RE0/jRj36ELVu2jCnkGQ6H4fP5AAA+nw+RSAQAEAgE0NjYWDivoaEBgUCgsgETUVXjnDMioim8/PLL8Hg8WLJkCd54441pz59JYc9du3Zh165dAID7778ffr9/1nEWw2Qylf01qjUORVFglZOwa7mPRatsgtfrhSRJEx63WCywqCLsmlR4vukeM/54/hyx9yzsdvuEzzvdc87mMRPFYcTfST3HweSMiGgKb731Fl566SW8+uqrSKfTSCQS2L59OzweD4LBIHw+H4LBINxuNwCgsbERQ0NDhccHAgE0NDRM+NwbNmzAhg0bCv8eHBws67X4/f6yv0a1xiEIAlLJ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
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We see that the majority of fighters are between the 25-35 range. This should be unsurprising, because it takes most fighters a few years of fighting amateur fights when they are in their younger twenties and many fighters start to retire after 35+. The career span of athletes to begin with are relatively short, almost always less than twenty years. The career span of combat sports athletes are even shorter because of the physical strain on their bodies and the damage they take in the cage/ring.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[30]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">df_final</span><span class="p">[</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;winner&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">][</span><span class="s1">&#39;f1_age_when_fight&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 1 Wins by Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Wins&#39;</span><span class="p">)</span>

<span class="n">bar</span> <span class="o">=</span> <span class="n">df_final</span><span class="p">[</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;winner&#39;</span><span class="p">]</span> <span class="o">==</span><span class="mi">1</span><span class="p">][</span><span class="s1">&#39;f2_age_when_fight&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 2 Wins by Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Age&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Wins&#39;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtcAAAIACAYAAABJtUDKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABA4klEQVR4nO3deXxU9b3/8fdMgmyRkBC2AC4xIKAgSgABJSJRq6U0N9fihgtF1LoCYrFahVvkGquAcAUVRCsX2ooLKVYp/iIaKqDGWhQBkVBBUQSyEBYJhOT7+4ObaYYszAnfmTMzeT0fDx5kznzmnM85OfOZT77nOzMeY4wRAAAAgJPmdTsBAAAAIFrQXAMAAACW0FwDAAAAltBcAwAAAJbQXAMAAACW0FwDAAAAltBcw8+UKVOUmprq6DF/+MMfFBsbG6SMwldDjpVT27Ztk8fj0QcffBDU7QCITNTswFGzESo0143QLbfcIo/HU+Pfn//8Z02cOFEffvih9W1+8MEH8ng82rZtm/V1Vzdu3DgNGDBALVq0COjFY+vWrfJ4PFq+fLnf8nvvvbfO5aeddpokBe1Yhdprr72mmJgYZWVluZ0KgFpEa81ev369brzxRp1xxhlq1qyZzjzzTI0bN0579+6t8zHUbGp2JKC5bqQuvvhi7dy50+9fZmam4uLilJSU5HZ69Tpy5Eid91VUVOj666/XnXfeGdC6zjrrLJ1xxhl69913/ZavXLlSp512Wq3LMzIyJCkijlUg5s2bp0mTJmn58uX64Ycf3E4HQC2isWZ/+umniouL0wsvvKCNGzfqueee05tvvqnrrruuznVRs6nZEcGg0bn55pvNsGHDar1v8uTJ5qyzzvJbNnPmTNOpUyfTvHlzc/nll5uFCxcaSebbb781xhjz0ksvmZiYGPPBBx+Y888/3zRv3tykpaWZTz75xBhjzNdff20k+f1LT0/3rf9Pf/qTOe+880zTpk3N6aefbsaPH28OHDjguz89Pd388pe/NL/97W9Nhw4dTFJS0gn3sSqnQIwZM8b06dPHd3vXrl3G6/Wa559/vsZySWbx4sW1Hquq2zk5Oebss882LVq0MJdccokpKCjwxZSWlppbbrnFtG/f3pxyyimmc+fOZvz48XXmVnXsFi5caC699FLTrFkzc8YZZ5hFixb5YoYMGWLGjh3r97jKykqTkpJiJk+eXO++b9261TRt2tTs2bPHXHnllWbatGk1Yj799FMzYMAA07RpU9O1a1fz6quvmtNPP91MnTrVF7N//35z7733muTkZNO8eXPTp08f8/rrr9e7bQCBaQw1u8prr71mPB6PKS0trTOGmk3NDnc0142Qk0L9+uuvm5iYGPP000+br776yrz00kumY8eONQq1x+MxF198sVm1apXZtGmTueyyy0xKSoopLy83R48eNX/5y1+MJPPxxx+bnTt3mqKiIt9jW7dubRYuXGi2bt1q8vLyTK9evcyoUaN8OaSnp5u4uDhz++23mw0bNpjPP//8hPvopLn+4x//aDwej9mzZ48x5tgLR58+fUxhYaGJiYnxWy7J7Ny5s9ZjNXnyZNOiRQtzxRVXmE8++cSsW7fO9OnTxwwZMsQXc88995jevXubDz/80Gzfvt2sXr3azJs3r87cqgp1x44dzaJFi8yXX35pHn74YePxeEx+fr4v/7i4OLN//37f43Jzc43X6zXbt2+vd98nTZpk/uM//sMYY8wrr7xizjzzTFNZWem7/+DBg6ZDhw5m+PDh5rPPPjNr1641AwcONM2bN/cV6srKSnPJJZeY9PR08/e//91s3brVPP/886ZJkyYmNzf3xL8AAPVqDDW7yoIFC0yLFi1MeXl5nTHUbGp2uKO5boRuvvlmExMTY1q2bOn7l5KSYoypWXwGDRrkVzSNOfbkPr5QSzL/+Mc/fDFr1641ksyXX35pjDHm73//u5Fkvv76a791nX766ebZZ5/1W5aXl2ckmeLiYmPMsULdtWtXU1FREfA+Ommuf/jhByPJLFmyxBhjzK233uobmTjvvPP8lp977rm+x9VWqGNiYszu3bt9y/70pz8Zj8djDh06ZIwxZsSIEebmm28OeD+qCvVvf/tbv+UDBw40N9xwgzHGmMOHD5ukpCQzf/583/3XXnutueqqq+pd95EjR0y7du1MTk6OMcaYsrIyk5CQYFasWOGLmTdvnmnZsqXZu3evb9mmTZuMJF+hfu+990zTpk39YowxZvTo0ebnP/95wPsKoHaNoWYbY8zOnTtN586dzf33319vHDWbmh3umHPdSA0YMEDr1q3z/Tt+nlqVjRs36sILL/RbNnDgwBpxHo9H5513nu92p06dJEm7du2qM4c9e/Zo+/btmjBhguLi4nz/rrzySklSQUGBL7Zv377yeoNzurZv317nnnuucnNzJR2bo3fppZdKki699FK/5VVz9+qSnJystm3b+m536tRJxhjt3r1bknTnnXfqtdde07nnnqv77rtPy5cvV2Vl5QlzPP6YDx48WBs3bpQknXLKKbrllls0f/58SVJRUZGWLl2qsWPH1rvOpUuXqrKyUldddZUkqWnTprr22ms1b948X8zGjRvVo0cPxcfH+5Z1795drVu39t3Oz8/XkSNH1KlTJ7/f46JFi7Rly5YT7huAE4v2mr17925dfvnl6t27tx5//PF6Y6nZ1Oxw1/g+iweSpObNmwf8kUQej+eEMV6vVzExMTUeU18Rqrpv1qxZGjp0aI37O3fu7Pu5ZcuWAeXaUBkZGXrzzTf1zTffaPv27RoyZIgkaejQoRo/fry++eYb/etf/9KwYcPqXc8pp5zid/v443DFFVfom2++0YoVK/T+++9r1KhR6tWrl959912/43cixhi/27fffrumT5+uzz//XCtXrlRiYqKGDx9e7zrmzZunwsJCNW/e3G+9MTEx2rVrl9q3b++3D3WprKxUfHy88vPza9x3/PEA0DDRXLN37Nihyy67TKmpqXrttdfUpEmTEz6Gmv3v9VKzww8j16hXz549tXbtWr9lDfkoo6onbEVFhW9Z+/bt1aVLF23evFmpqak1/jVr1uzkkndg2LBh2rp1q1588UWlpaWpVatWkqQhQ4Zo27ZtevHFFxUbG6v09PST3lZiYqKuu+46Pf/883rrrbeUl5fnG9Goy/HHfO3aterRo4fvdmpqqi699FLNnz9fL7zwgkaPHl3vRxEWFBRo5cqVWrp0qd9o2GeffaaUlBS99NJLko79/jdt2qTS0lLfYzdv3uz3UVlpaWnau3evysrKavwOqz4CC0BoRFrN3rp1qy6++GL17NlTb7zxhpo2bRrQ46jZ1Oxwxsg16nX//ffrmmuuUf/+/XXllVdqzZo1WrhwoaTARkeqnH766fJ6vXr77bd1zTXXqGnTpoqPj9e0adM0ZswYtW7dWpmZmWrSpIk2bdqk5cuX6/nnn3ecb0FBgQ4cOKBvvvlGkrRu3TpJxwpZXFxcnY9LT09XbGysZsyYobvvvtu3PD4+XhdccIFmzJih/v3769RTT3WcU3UPP/yw+vbtq3POOUder1eLFy9WXFzcCQvaggUL1L17d6WlpWnRokVau3atnn76ab+Y22+/XaNGjVJ5ebnefPPNetc3b948paSkKDMzs8Z9I0eO1AsvvKBJkybphhtu0KOPPqqbbrpJU6dO1aFDh3T//ferefPmvt//pZdeqoyMDGVlZemJJ57Qeeedp5KSEq1Zs0bNmjU74aVOAPZEUs3euHGjMjIy1Lt3b82ePVtFRUW++9q2bVvvyDA1+9+o2eGHkWvUKysrS7///e+VnZ2tXr16afHixZo8ebIkORqlaN++vR5//HFlZ2erY8eO+vnPfy5JuvHGG7VkyRK99dZb6t+/v/r166cpU6b45v85deutt+r888/X5MmTVVFRofPPP1/nn3++Pvnkk3ofd+qpp6pfv37av3+/b+5elaFDh2r//v0nnLsXiGbNmunRRx9V3759lZaWps8//1zLly/3mx9Xm+zsbM2bN0+9e/fWwoUL9fLLL6tfv35+MZmZmYqPj9dll12mM888s851HTlyRH/4wx/0i1/8otb7r7nmGm3dulXvvvuuWrRoobffflu7du1Sv379NGrUKI0bN05xcXG+37/H49GyZcuUlZWlCRMmqHv37vrpT3+qt956S2eddZbDIwTgZERSzV6yZIl27typFStWqHPnzurYsaPv37ffflvvY6nZ/0bNDj8ec/xEIOAEfve732nWrFl+owxwX3FxsTp16qRFixbpP//zP4O2ne3bt+uMM87QsmXL9LOf/Sxo2wFgBzU7PFGzoxfTQlCv8vJyTZ8+XVdddZVatmyp9957T08++aTuuusut1PD/ykvL9euXbs0depUJScn13rZ8GQsWrRInTp10plnnqnt27fr17/+tU4//XRdfvnlVrcD4ORRs8MfNTv60VyjXh6PR++//76mT5+u/fv368wzz9RDDz2kBx54wO3U8H9Wr16toUOH6swzz9TChQsdvYM9EEVFRZo8ebK+++47JSYmavDgwXr11VcDfuMRgNChZoc/anb0Y1oIAAAAYAlvaAQAAAAsobkGAAAALKG5BgAAACyJqjc0fv/99zWWJSUlqbCwMKDHByPW7e1HWqzb24/mWLe3H82xNtaZnJwc0OOjzfF1O1i/y2CuOxLzIOfGlQc524+tr2Yzcg0AAABYQnMNAAAAWEJzDQAAAFhCcw0AAABYQnMNAAAAWEJzDQAAAFhCcw0AAABYQnMNAAAAWEJzDQAAAFhC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NQAAAGBJg5vrL774Qhs3brSZCwAgSKjZABAasYEGTp48Wdddd526d++unJwcvfXWW/J6vbriiiuUlZUVzBwRRirGjvD9vOv//o+Zv8ydZADUiZoNAO4IeOT622+/Vbdu3SRJ7777riZPnqxp06bp//2//xe05AAADUPNBgB3BDxybYyRJP3www+SpM6dO0uSDh48GIS0AAAng5oNAO4IuLk+++yz9eKLL6qkpET9+vWTdKxon3rqqUFLDgDQMNRsnCymAQINE/C0kLvuukstWrTQ6aefrpEjR0qSvv/+e1111VVBSw4A0DDUbABwR8Aj16eeeqquv/56v2UXXHCB9YQAACePmg0A7gi4uT569Kjef/99bdu2TWVlZX733X333dYTAwA0HDUbANwRcHP9zDPPaPv27erbt6/i4+ODmRMA4CRRswHAHQE315999pmeeeYZtWzZMpj5AAAsoGYDgDsCfkNjUlKSysvLg5kLAMASajYAuCPgkeshQ4boySef1JVXXqnWrVv73XfuuefazgsAcBKo2QDgjoCb67/97W+SpD/96U9+yz0ej5555hm7WQEATgo1GwDcEXBzPWfOnGDmAQCwiJoNAO4IeM41AAAAgPrVO3I9fvx4zZw5U5L0q1/9qs64Z5991m5WAADHqNkA4L56m+vbb79da9asUc+ePXXPPfeEKicAQANQswHAffU21927d9d9992nH374QR06dFCPHj3Us2dP9ejRQ23btg1VjgCAAFCzAcB9J3xD46xZs7R3715t2rRJmzZt0ptvvqm5c+cqMTHRV7iHDRsWilwBACdAzQYAdwX0aSGtW7fWwIEDNXDgQEnSwYMHlZubq7/+9a/64IMPKNQAEEao2QDgnoCaa2OMtm3bpk2bNmnjxo366quvlJCQoIEDB6pHjx7BzhEA4AA1GwDcc8LmOjs7W19//bWSk5N19tlnKyMjQ3fddZeaN28eivwAAA5QswHAXSf8nOvvv/9esbGxatu2rTp06KAOHTpQpAEgTFGzAcBdJxy5nj17tt+bY9566y3t379fZ599tnr06KHu3bvrjDPOCEGqAIAToWYDgLtO6g2Nr7/+uvbt26dXXnklqEkCAAJHzQYA9zToDY2bN2/WwYMHddZZZ2no0KHBzhEA4AA1GwDcc8Lm+vHHH9dXX32lo0ePKjU1VT179tRPfvITdevWTaecckoocgQABIiaDQDuOmFz3aNHD2VlZemss85SbGxAA90AAJdQswHAXSesvJmZmSFIAwBgAzUbANx1wo/iAwAAABAYmmsAAADAEpprAAAAwBKaawAAAMASmmsAAADAEj6nCQBQw9y5c/Xpp58qPj5e06dPlyQtWbJE7777rlq1aiVJuu6663TBBRdIkpYuXaqVK1fK6/Vq9OjR6tOnj1upA4CraK4BADVccskl+slPfqI5c+b4Lf/pT3+qESNG+C3bsWOH1qxZoxkzZqikpERTp07VrFmz5PVycRRA40PlAwDU0LNnT8XFxQUUm5+fr0GDBqlJkyZq166dOnTooIKCgiBnCADhKSQj11xeBIDosGLFCq1atUopKSm66aabFBcXp+LiYnXt2tUXk5iYqOLiYhezBAD3hKS55vIiAES+yy+/XFdffbUk6ZVXXtHChQt15513yhgT8Dpyc3OVm5srScrOzlZSUpJ21RGblJRU6/La4p3E1hUfrNi64kOds4086oqtLjY2NqC4YMaSR8NjwyWPSMzZ9xhH0Q3Us2dP7d69O6DYui4vduvWLchZwraKsf/+w6mqSMfMX1ZvbPViXlcsAHe0bt3a9/OwYcP0xBNPSJLatGmjoqIi333FxcVKTEysdR0ZGRnKyMjw3S4sLKxze/XddzKxwVw3OR9rwANdZ7BiyaPhseGSR7jnnJycXOdjXH1D48leXqxtBEQK3l/mbseGwwiM28egunD4KzWSYt3efjTHBnsUJFyUlJQoISFBkvTxxx+rS5cukqS0tDTNnj1bw4cPV0lJiXbu3KnU1FQ3UwUA17jWXNu4vBgOIyDEurf9cPhrOZJi3d5+NMcGexTEDU8//bQ2btyo/fv364477tDIkSO1YcMGbdu2TR6PR23bttVtt90mSerSpYsGDhyoCRMmyOv1asyYMUzlA9BoudZc27i8CAAIjnHjxtVYdumll9YZn5WVpaysrCBmBACRwbWhhZKSEt/Px19eXLNmjcrLy7V7924uLwIAACBihGTkmsuLAAAAaAxC0lxzeREAAACNAUPCAAAAgCU01wAAAIAlNNcAAACAJTTXAAAAgCU01wAAAIAlNNcAAACAJTTXAAAAgCU01wAAAIAlNNcAAACAJTTXAAAAgCU01wAAAIAlsW4nAAAAIlvF2BG+n3dVWx4zf1nokwFc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TB999JE2btzoW6fTWCf75SRfJ8c20OMlBX4uBHrOSs7OmWCd44HGOjlWTvbLyb4F67ydOHGinnrqKVVUVOiOO+7Q888/L6/XK2OMHnjgAT311FO+WCcvzI2Rk3PPyXPFyfkvhUftrso70OdNsGqik+eC03gneTjZv2Ae52CdG1Lwzulg1PSG5OHkODs5N4L5HHR6/tclsOuXEWLPnj2+v/aGDx+u3/zmN7r66qt15513asKECb4Tdffu3Xr44Yf9Htu6dWtlZmbqvffe81v+m9/8xu8v5OoOHjzod7uiokLnn3++pGN/dV944YWSpF69eul///d//WJ37dqliRMnSpL69++vN954Q7/73e9q/cXFxcXppptu0k033aRNmzZp9erVmjRpkjp37qzBgwcrIyPDL4eKigrFxMToyJEjSk1NlSQlJyervLzcb72lpaW68sorJUkrVqzwfa3nlVde6fcFFzfccIPWr1+vG2+8UaeddpqkYyfsnDlzaj0uVbZu3aonn3xS0rHfR15eXoNjneyXk3ydHNtAj1dVvoGcC4Ges5KzcyZY53igsU6OlZP9crJvwTpvjTE6evSoysrKdPjwYf3444+Ki4tTeXm5Kioq/GJfeOEF3wvzI488optvvlmPPPKI1q9frxdeeMHvhbkxcnLuOXmuODn/pfCo3ZLzGhOMmlidkxoeSLyTPJzsX7CPczDODSl453QwanpD8nBynJ2cG8F8Dlbn9PyvLqqa66ZNm+rLL79U9+7d9cknn/guI1SNKlVp27at/vKXvyg9Pb3GX17H/+XUuXNn3XbbberYsWON7f3qV7/yu92kSRN99tln+vHHH+XxePTxxx+rf//+2rhxY41PBzh69KgqKyt9y7OyspSYmKjJkyfX+Ja36nr06KEePXrol7/8pT7//HOtWbPG7+S44oorfN8Id9555+kPf/iD+vfvry+++EJnnHGG37qqH5P09PQ67xsxYoQGDx6sl19+WW3atNHIkSPr/Kai0tJS/fWvf5UxRocOHZIxxhd7/EUSJ7FO9stJvtWd6NjWd7wqKyv9bgd6LgR6zkrOzplgneOBxgZ6bjndLyf7FqzzdujQoRo3bpwqKyt17bXXasaMGWrXrp22bNmiQYMG+cU6eWFujJyce06eK07Ofyk8arfk7HkTrJro5LngNN5JHk72r7qTPc4NreWS83MjWOd0MGp6Q/Jwcj47OTeC+Rx0ev7XJaqa61tvvVXPP/+8du7cqS5duvhOjH379umKK67wxY0bN045OTmaMmWKSktLJR37y6tv374aP3683zp/8Ytf1HlAR48e7Xd77NixWrx4sTwejx5++GG98847mjt3rhITE3X77bf7xfbt21dffPGFevfu7VtWdZnlxRdf9Iut7aT3er3q06dPjS+huPLKK3XaaafpnXfe0c6dO1VRUaGdO3eqX79+ysrK8otNS0tTWVmZmjVrpmuvvda3/IcffqixzTZt2mjChAn65JNP9Nhjj+nw4cO1HpNhw4bp0KFDko49mfbv369WrVpp7969NQqik1gn++UkXyfHtr7jlZyc7Bdb27nw7LPPKiEhQbfddpsvLtBzVnJ2zgTrHA801sm55WS/nO5bMM7b4cOH+5roxMREpaena/369crIyPCNrlVx8sLcGDk595w8V5ycI1J41G7J2fMmWDXRyXOhIfGB5uFk/2we54bWcsl5LQvWOR2Mmt6QPJycz1Lg50Ywn4NOz+c6Bfy5IhHi22+/NZ999pk5dOiQ3/J//vOffre3bNlitmzZYowx5ptvvjHLli0z//jHP2pdZ/XYb7/91rz55pt1xn777bfm888/P+H2na53x44dAa/XSWygOVSP2759u3nttdes5OrkGDj5nVXP4fDhw2b79u1WcnCyb1999VVA+VaPa+j2P/3001rjq5s9e3ad9znZr+P9z//8zwljnMQFErtjx46AnufVHT582EyfPt1aDoH4+uuvzWOPPWamTZtmduzYYV588UVz8803m/Hjx5svv/zypNcfDZzW10B/707qRdW63a7dwVx3sGr4yeTspDbbfJ12uo+B1vL61ltXjXZS/4/nVk1vSP0NdN1Vglmvndb2hrwWRNUbGt9++2298847Sk5O1vbt23XLLbeoX79+kvwnqb/66qtat26dKioq1Lt3bxUUFKhnz55av369zjvvPL+/io+P3bJli84555xaYwPdvtP1Ll++XH/729/UqVOnE6737bff1ooVKwKKDTQHJ8fLSa5OjoGTHIJxDE523+rKN1jbP/6NM5L0xRdf6Nxzz/XFN+R4Hb9eY4w2bNhQY72BxjmNdZKvk2PgJNaW9957T0OHDrW+3kgSqvpaX7042XXbes425Hg0tM7ZquHhkrOTHJzuY7gcO7drerDXfTL12ubri7XXAsfteBibMGGC7y+pXbt2mUmTJpm33nrLGGPMAw884BdXUVFhysrKzE033WQOHjxojDn2l9L9999fY51OYgPZfrDXazuHcDkG4ZBDpBzbX//612bWrFnmiy++MBs2bDBffPGFGTt2rNmwYYPZsGFDg9f7wAMPBLTeQOOcxjrJ18kxcJqDDXfccUdQ1htJ3H7+BXvdTtYbLscjUnOO9t+32zU92OsOVr12un+2XguiatJfZWWlmjVrJklq166dpkyZon/+8596+eWX/eYUxcTEyOv1qmnTpmrfvr1atGghSTrllFNqTKB3Ehvo9oO53mDkEA7HIBxyiKRj+/jjjyslJUVvvPGGWrRooXPOOUennHKKevbsWeOd4U7Wm52dHdB6A41zGuskXyfHwGkOgZo4cWKt/+6//37ffMXGzO3nX7DX7WS94XI8IjHnxvD7drumB3vdwarXTvfP1mtBVDXXrVu31rZt23y3mzVrpgcffFD79+/XN99841seGxvrmyhf/bMyf/zxxxpvMnISG+j2g7neYOQQDscgHHKIpGPr9Xo1fPhw3XnnnXrjjTe0YMGCGh8TF8z1Otm+k1gn+QYzh0CVlpbq7rvv1qRJk2r845vm3H/+BXvdTtYbLscjEnNuDL9vt2t6sNcdDrENia9TwGPcEaCwsNCUlJTUet+mTZt8Px85cqTWmNLSUt+bKxoSG+j2g7neYOQQDscgHHKIpGN7vH/84x9m8eLFtd4XrPU2JC6Q2IbmazOHQM2dO7fOnJ5++umTXn+kc/v5F+x1Oz1Xw+F4RGLOjeH3fTy3anooXi/CJbYh8VWi6g2NAAAAgJuialoIAAAA4CaaawAAAMASmmsAAADAEpprQNKUKVM0evRolZeXu50KAOAEqNkIZzTXaPR2796tTZs2SZI++eQTl7MBANSHmo1wF+t2AoDbVq1apW7duik1NVV5eXkaOHCgJGn//v2aM2eONm3apOTkZJ133nnasGGDpk6dKkn67rvv9OKLL+pf//qXWrVqpWuuuUaDBg1yc1cAIOpRsxHuGLlGo5eXl6eLLrpIF198sT777DPt3btXkrRgwQI1a9ZM8+bN01133aW8vDzfY8rKyvTYY4/poosu0gsvvKD77rtPCxYs0LfffuvSXgBA40DNRrijuUaj9uWXX6qwsFADBw5USkqK2rdvrw8++ECVlZX66KOPNHLkSDVt2lSdO3dWenq673Gffvqp2rZtq6FDhyomJkYpKSkaMGCAPvzwQxf3BgCiGzUbkYBpIWjU3n//ffXu3VutWrWSJF100UW+UZGKigq1adPGF1v95z179mjLli265ZZbfMsqKio0ZMiQkOUOAI0NNRuRgOYajdaRI0e0du1aVVZWauzYsZKko0eP6uDBg9q7d69iYmJUVFSk5ORkSVJRUZHvsW3atFHPnj31yCOPuJI7ADQ21GxECpprNFoff/yxvF6vpk+frtjYfz8VZs6cqVWrVql///569dVXdccdd6iwsFB5eXlKSkqSJPXt21d//OMftWrVKt8bYrZt26ZmzZqpc+fOruwPAEQzajYiBXOu0Wjl5eVp6NChSkpKUuvWrX3/rrjiCv3973/XmDFj9OOPP+q2227TM888o8GDB6tJkyaSpObNm+u3v/2tVq9erdtvv1233XabFi9erKNHj7q8VwAQnajZiBQeY4xxOwkgEixatEh79+7V3Xff7XYqAIAToGbDLYxcA3X47rvvtH37dhljVFBQoPfee0/9+/d3Oy0AQC2o2QgXzLkG6nDo0CHNmjVLJSUlio+P1/Dhw9WvXz+30wIA1IKajXDBtBAAAADAEqaFAAAAAJbQXAMAAACW0FwDAAAAltBcAwAAAJbQXAMAAACW0FwDAAAAlvx/Javd3bdP4NgAAAAASUVORK5CYII=
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As is consistent with the age distribution, the prime year range of 25-35 is located to the leftmost side of the x-axis with the most wins.</p>
<p>Moving onto physical attributes, let's see what results from analyzing height.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[31]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">histplot</span><span class="p">(</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;height_f1&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">kde</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 1 Heights&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">histplot</span><span class="p">(</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;height_f2&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">kde</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 2 Heights&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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JXZs2bJl2LBhA7Zu3QqHw4HVq1cDACoqKrBkyRKsXr0aoihi1apVXKlJREREWSFlxZnBYMAvf/nLAcfy8vKwZs2aIc9fvnw5li9fnopoRERERKqR8tWaRETphJtoE1GqsTgjIhoBN9EmolRja0FEFCduok1EqcCeMyKiOHETbRrKUBtpcxNtmggWZ0REcUj0JtpA+m2kreZsgLo20o53E22AG2mPRs3ZgOTkY3FGRBSHRG+iDaTfRtpqzgaoayPteDfRjvfxyabmn62aswHjzzfSJtqcc0ZEFAduok1EqcKeMyKiUXATbSJKJRZnRESj4CbaRJRKvJwjIiIiUhEWZ0REREQqwuKMiIiISEVYnBERERGpCIszIiIiIhVhcUZERESkIizOiIiIiFSExRkRERGRirA4IyIiIlIR3iGA0kogKsAXkQYci0rDnExERJSGWJxRWvFFJGw+7Bxw7Ooqh0JpiIiIEo/DmkREREQqwuKMiIiISEVYnBERERGpCIszIiIiIhVhcUZERESkIizOiIiIiFSExRkRERGRirA4IyIiIlIRFmdEREREKsI7BFBWGer2T2atCKNGVigRERHRQCzOKKsMdfunG2Y6YNQICiUiIiIaiMUZERFRGmDPf/ZgcUZERJQG2POfPbgggIiIiEhFWJwRERERqQiLMyIiIiIVYXFGREREpCIszoiIiIhUhMUZERERkYqwOCMiIiJSERZnRERERCrC4oyIiIhIRVicEREREakIb99ERER0nqHuYwnwXpaUGizOKGl4k14iSldD3ccS4L0sKTVYnFHS8Ca9REREY8c5Z0REREQqwuKMiIiISEVSNqzZ29uLn/zkJzh16hQEQcDXv/51lJaWYsOGDejo6EBhYSHuu+8+5ObmAgA2bdqErVu3QhRF3HnnnViwYEGqohIREREpJmXF2YsvvogFCxbgO9/5DiKRCILBIDZt2oR58+Zh2bJlqKurQ11dHVasWIHm5mZs374dTz/9NNxuN9atW4dnn30WosiOPiIiIspsKal2fD4fDh48iGuvvRYAoNVqkZOTg/r6etTU1AAAampqUF9fDwCor6/H0qVLodPpUFRUhJKSEjQ0NKQiKhEREZGiUtJz1t7ejvz8fDz33HNoampCZWUlVq5cie7ublitVgCA1WqFx+MBALhcLlRXV8ceb7PZ4HK5Bj3vli1b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
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>The heights still appear to be normally distributed, but not as smoothly as the ages. The majority of the data is contained with the 170-190cm range. For reference, 170cm is about 5'7" and 190cm is about 6'3". While this distribution is higher than what it would be for the average population, for professional athletes it makes perfect sense because an increase in height brings many physical advantages.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[32]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">df_final</span><span class="p">[</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;winner&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">][</span><span class="s1">&#39;height_f1&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 1 Wins by Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Wins&#39;</span><span class="p">)</span>

<span class="n">bar</span> <span class="o">=</span> <span class="n">df_final</span><span class="p">[</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;winner&#39;</span><span class="p">]</span> <span class="o">==</span><span class="mi">1</span><span class="p">][</span><span class="s1">&#39;height_f2&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Fighter 2 Wins by Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Height&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Wins&#39;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtcAAAJZCAYAAABiNBn1AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABrfklEQVR4nO3de1yUdfr/8fcMaJiIclARw1NoaWpmWKmVmZRbWUt2Puea2qqVUWZFpW22aaWmZbXpdjCtzQ7SYTutWfg1LKnW0rRWKHVVFDkIoqIC9+8Pf8yCMOPw4WaGYV7Px8PHQ2bmmvua+3Bxcc/nc98Oy7IsAQAAAKg3p78TAAAAAJoKmmsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcx0kpk2bpoSEhDrFvPrqqwoNDW2gjBovk3VVV5s3b5bD4dCqVasadDn19dVXX8nhcGjbtm1exwTKZwMaM2q296jZ/0PNbhxorpuQW2+9VQ6Ho8a/f/zjH7r33nv1zTff2L7MVatWyeFwaPPmzba/d1WTJk3SmWeeqeOPP96rXx7Z2dlyOBz65JNPqj1+5513un28U6dOktRg68pXunTpounTp9d43KToDho0SDk5OYqLi7MzRUlSaGioXn31VdvfFwgUTbVmr1u3TjfddJO6dOmisLAwde3aVZMmTdKePXvcxlCzqdlNCc11E3POOecoJyen2r/k5GSFh4crJibG3+l5dOjQIbfPlZeX6/rrr9f48eO9eq8TTzxRXbp00RdffFHt8RUrVqhTp061Pp6UlCRJAbGufKV58+aKjY2V00mpABpCU6zZP/zwg8LDw7Vw4UJt2LBBL774oj788ENdd911bt+Lmm0PanbjwNpvYioPrKr/wsLCav3a7JlnntEJJ5yg448/XsOHD9frr79e61/JX3/9tfr376/jjz9eAwYM0Pfffy/pyFdJ55xzjiSpa9eucjgcOu+881xx//jHP9SvXz+FhYWpS5cuSklJ0b59+1zPn3feeRo9erQefvhhdejQQR07dnT7uZ599lnddddd6t27t9frYtiwYdUKcm5urjZu3KjU1NQaj//888+uQn30uqr8+f3339fJJ5+sli1baujQocrOzna9pri4WKNGjVJsbKyOO+44xcfHKyUl5Zg5/v777xo2bJhatGihrl27asmSJa7nhgwZorFjx1Z7vWVZOvHEEzVt2jSv14MnWVlZuuKKK9SmTRtFRkbqwgsv1Lp161zP13bmZPny5erTp4/CwsLUt29fpaeny+FwaPHixdXee8eOHbr00kt1/PHHq1u3bnr99dddz3Xp0kXl5eUaNWqU62wdEIyaYs2+5ZZb9MILLygpKUndunXT8OHD9eSTT+qzzz5TcXGx23VBzT42anZgoLkOUu+9957uvfdeTZ48WT/++KOuu+46TZkypcbrKioq9MADD2ju3Ln64YcfFBkZqauvvlplZWWKj4/X+++/L0las2aNcnJy9N5770k6Mvbvz3/+s+655x5t2LBBixYt0vLly3X77bdXe/+lS5dq9+7d+uKLL7RixQpbP+OwYcP0448/Ki8vT9KRMx19+/bVFVdcoXXr1lV7XJLOP/98t++Vk5OjF154QUuWLFFGRob27NmjP/3pT67nH3roIf3www96//33tWnTJr311lvq2bPnMXOcMmWK/vSnP2nt2rW64YYbdNNNN+m7776TJN1+++168803VVJS4nr9ihUrtHnz5mrLNrVr1y6dffbZateunf7v//5P33zzjU466SSdd9552r17d60x27dv12WXXaYzzzxTP/zwg+bMmeP2F9L999+vm266ST/99JOuvvpqjRo1Sps2bZIkZWZmKiQkRM8884zrbB0A9wK9ZhcVFalFixY6/vjj3b6Gmu0ZNTuAWGgybrnlFiskJMRq2bKl61+3bt0sy7KsqVOnWieeeKLrtYMGDbJuvPHGavFTpkyxJFn//e9/LcuyrFdeecWSZH3//feu16xevdqSZP3yyy+WZVnW//3f/1mSrN9//73ae3Xu3Nl64YUXqj2Wnp5uSbIKCgosy7KsIUOGWN27d7fKy8u9/oyvvPKKFRIS4tVrd+7caUmyli5dalmWZd12223W3XffbVmWZZ166qnVHu/du7cr7uh1NXXqVCskJMTKzc11Pfbmm29aDofDOnDggGVZlnXZZZdZt9xyi9ef4/fff7ckWQ899FC1xwcOHGjdcMMNlmVZ1sGDB62YmBhrwYIFruevvfZa6+KLL/b43p07d7aaN29ebT9o2bKlFRYWVm37Tp061TrzzDOrxVZUVFjdunWz5syZY1mWZX355ZfVYh588EGrc+fOVllZmSvmk08+sSRZr7/+erXPNmvWLNdrDh8+bLVs2dJ68cUXXY+FhIRYr7zyijerC2iSgqFmW5Zl5eTkWCeccIJ1zz33eHwdNZua3VRw5rqJOfPMM7V27VrXv6PHqVXasGGDzjrrrGqPDRw4sMbrHA6HTj31VNfPlV8D7tq1y20Ou3fv1pYtW5SSkqLw8HDXv4suukjSka+1Kp1++ukNNjasffv26t27t5YvXy7pyBmEyjMd559/frXHK79edCcuLk5t27Z1/dyxY0dZlqXc3FxJ0vjx4/XOO++od+/euuuuu/TJJ5+ooqLimDkevc4HDx6sDRs2SDrydfGtt96qBQsWSJLy8/O1bNkyjRkz5pjvO2HChGr7wdq1a7Vw4cJqr8nMzNT3339fbRu1atVKmzdvdp2tONqGDRs0YMAAhYSEuP0Mlfr16+f6f2hoqNq3b+9xvwGCUVOv2bm5ubrwwgvVt29fPfHEEx5fS82mZjcVwXfNniauRYsWXl+SyJsxU06ns9pBWRnjqQhVPjd37lwNHTq0xvMnnHCC6/8tW7b0KldTSUlJ+vDDD7V161Zt2bJF5557riRp6NChuvvuu7V161b99ttvGjZsmMf3ad68ebWfj14Pw4cP19atW/XZZ5/pq6++0o033qg+ffroiy++qLb+jsWyrGo/jxs3TrNmzdJPP/2kFStWKCoqSiNGjDjm+0RFRdXYD44el1lRUaFhw4bpueeeqxHfunVrt+999H7jbj+qbZ1588sLCCZNuWZv27ZNF1xwgRISEvTOO++oWbNmx4yhZv8PNTtwceY6SPXq1UurV6+u9pjJpYwqD8by8nLXY+3bt1d8fLx+/fVXJSQk1PgXFhZWv+TrYNiwYcrOztbLL7+sxMRERURESJLOPfdcbd68WS+//LJCQ0M1ZMiQei8rKipK1113nf72t7/pn//8p9LT011nNNw5ep2vXr262ri/hIQEnX/++VqwYIEWLlyoUaNG2XYd28TERP3888/q2LFjjW1U9YxPVb169VJmZma17X30fuSt5s2bV3sfAO4FWs3Ozs7WOeeco169eum9997Tcccd51UcNds9anbgoLkOUvfcc4/+8Y9/6Nlnn1VWVpYWLVqkRYsWSfLu7Eilzp07y+l06uOPP1Zubq6KiookSY8//rjmzZun6dOna/369fr111+VlpamcePGGeWblZWltWvXauvWrZLk+tqs6sSR2gwZMkShoaGaPXt2tckvrVu3Vv/+/TV79mydccYZatWqlVFelVJTU/Xee+/p119/1aZNm7RkyRKFh4e7rsPqzt///ne98cYb+s9//qNHHnlEq1ev1qRJk6q9Zty4cXrppZe0YcMG3XbbbfXKs6qJEyeqvLxcycnJ+r//+z9t3rxZq1atUmpqqjIyMmqNGT9+vHbt2qU///nP2rhxo7788kulpqZKqtt+Ix25WsGXX36pHTt2uCYqAahdINXsDRs26JxzztFJJ52kefPmKT8/Xzt37tTOnTuP2ZxRs92jZgcOmusgNXLkSD355JOaMWOG+vTpoyVLlmjq1KmSVKezFO3bt9cTTzyhGTNmqEOHDvrjH/8oSbrpppu0dOlS/fOf/9QZZ5yhAQMGaNq0aR4vt+fJbbfdptNOO01Tp05VeXm5TjvtNJ122mmuWdrutGrVSgMGDNDevXtrzCwfOnSo9u7de8yxe94ICwvTI488otNPP12JiYn66aef9Mknn3j8qk6SZsyYoZdeekl9+/bVokWL9Nprr2nAgAHVXpOcnKzWrVvrggsuUNeuXeuda6X27dtr9erViomJ0ciRI3XSSSfphhtu0JYtW9ShQ4daYzp27KgPPvhAGRkZ6tevn+666y7XzQ/qenZr1qxZ+v7779W1a1e3Z10AHBFINXvp0qXKycnRZ599phNOOEEdOnRw/fvvf//rMZaa7R41O3A4rKMHDCFo/eUvf9HcuXOVn5/v71RQRUFBgTp27KjFixfriiuu8Hc6NaxcuVJDhgzRTz/9pD59+vg7HSBoULMbJ2o2mNAYpA4fPqxZs2bp4osvVsuWLfXll1/qqaee0oQJE/ydGv6/w4cPa9euXXrssccUFxen5ORkf6ckSXrhhRd06qmnKi4uThs2bNDdd9+tM888kyINNCBqduNHzUYlmusg5XA49NVXX2nWrFnau3evunbtqgcffFCTJ0/2d2r4/77++msNHTpUXbt21aJFi+o0g70hbdmyRU888YR27dql2NhYXXDBBZo5c6a/0wKaNGp240fNRiWGhQAAAAA2YUIjAAAAYBOaawAAAMAmTWrM9Y4dO2p9PCYmxuiajE05LhByDJS4QMgxUOICIceGiouLi6vz+zUFtdXtxrRdAj0uEHIMlLhAyLGpxzWmHD3VbM5cAwAAADZpUmeuAQD2qaio0P3336+oqCjdf//9Kikp0Zw5c7R79261bdtWd999t8LDwyVJy5Yt04oVK+R0OjVq1Cj169fPv8kDgJ9w5hoAUKuPP/642h360tLS1KdPH82bN099+vRRWlqaJGnbtm3KyMjQ7NmzlZqaqr///e+qqKjwU9YA4F801wCAGvLz8/XDDz9o2LBhrscyMzM1ZMgQSdKQIUOUmZnpenzQoEFq1qyZ2rVrp9jYWGVlZfklbwDwN4aFAABqePXVV3XjjTfqwIEDrseKiooUGRkpSYqMjFRxcbGkI7d77t69u+t1UVFRKigoqPV9ly9fruXLl0uSZsyYoZiYmBqvCQ0NrfXxYyHOv8tq6nGBkGNTjwuEHCWaawDAUb7//nu1bt1a3bp1088//3zM19flXmRJSUlKSkpy/VzbTPzGdEWAQI8LhBwDJS4QcmzqcY0pR09XC6G5BgBU8+uvv+q7777Tv//9bx06dEgHDhzQvHnz1Lp1axUWFioyMlKFhYWKiIiQJEVHRys/P98VX1BQoKioKH+lDwB+xZhrAEA1119/vV588UXNnz9fkyZNUu/evXXnnXcqMTFR6enpkqT09HQNGDBAkpSYmKiMjAwdPnxYubm5ysnJUUJCgj8/AgD4DWeuAQBeSU5O1pw5c7RixQrFxMQoJSVFkhQfH6+BAwcqJSVFTqdTo0ePltPJuRsAwYnmGgDg1imnnKJTTjlFktSqVSs98sgjtb5u5MiRGjlypC9TA4BGiVMLAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE2a7B0ay8dc5vr/rqOeC1nwgW+TAQB4RM0G0FRw5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATUJ9sZBDhw5p6tSpKisrU3l5uc466yxdffXVWrp0qb744gtFRERIkq677jr1799fkrRs2TKtWLFCTqdTo0aNUr9+/XyRKgAAAGDMJ811s2bNNHXqVIWFhamsrEyPPPKIq1m+5JJLdNlll1V7/bZt25SRkaHZs2ersLBQjz32mObOnSunkxPtAAAAaLx80q06HA6FhYVJksrLy1VeXi6Hw+H29ZmZmRo0aJCaNWumdu3aKTY2VllZWb5IFQAAADDmkzPXklRRUaEpU6Zo586dGj58uLp3765///vf+uyzz7Ry5Up169ZNN998s8LDw1VQUKDu3bu7YqOiolRQUFDjPZcvX67ly5dLkmbMmKGYmBjXc7s85FL1dZ6EhoZ6/dpAiwuEHAMlLhByDJS4QMjRH3EAgMDhs+ba6XTqqaee0r59+/T0009r69atuvDCC3XllVdKkt566y0tWrRI48ePl2VZXr1nUlKSkpKSXD/n5eV5Feft62JiYrx+baDFBUKOgRIXCDkGSlwg5NhQcXFxcXV+P0jlY/43rPDokyohCz7wbTIAID9cLaRly5bq1auX1q5dqzZt2sjpdMrpdGrYsGHKzs6WJEVHRys/P98VU1BQoKioKF+nCgAAANSJT5rr4uJi7du3T9KRK4esW7dOHTt2VGFhoes1a9asUXx8vCQpMTFRGRkZOnz4sHJzc5WTk6OEhARfpAoAAAAY88mwkMLCQs2fP18VFRWyLEsDBw7U6aefrmeffVabN2+Ww+FQ27ZtNXbsWElSfHy8Bg4cqJSUFDmdTo0ePZorhQCAj3D5VAAw55PmunPnznryySdrPH7HHXe4jRk5cqRGjhzZkGkBAGrB5VMBwByVDwBQDZdPBQBzPrtaCAAgcDTE5VMl95dQNb18KpddbTzLaupxgZBjU48LhBwlmmsAQC0a4vKpktklVE0ue1iXuMZ0SUa74wIhx0CJC4Qcm3pcY8rR0+VTGRYCAHCLy6cCQN3QXAMAquHyqQBgjmEhAIBquHwqAJijuQYAVMPlUwHAHM01ACDolI/537W6j77iSMiCD3ybDIAmhe/tAAAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiECY1HYZILAAAATHHmGgAAALAJzTUAAABgE5prAAAAwCaMubZB1XHaUvWx2ozTBgAACB6cuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALBJqL8TCGblYy6r9vOuKv8PWfCBb5MBAABAvXHmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADYJ9XcCqLvyMZdV+3lXlf+HLPjAt8kAAADAhTPXAAAAgE18cub60KFDmjp1qsrKylReXq6zzjpLV199tUpKSjRnzhzt3r1bbdu21d13363w8HBJ0rJly7RixQo5nU6NGjVK/fr180WqAAAAgDGfNNfNmjXT1KlTFRYWprKyMj3yyCPq16+f1qxZoz59+ig5OVlpaWlKS0vTjTfeqG3btikjI0OzZ89WYWGhHnvsMc2dO1dOJyfaAQAA0Hj5pFt1OBwKCwuTJJWXl6u8vFwOh0OZmZkaMmSIJGnIkCHKzMyUJGVmZmrQoEFq1qyZ2rVrp9jYWGVlZfkiVQAAAMCYzyY0VlRUaMqUKdq5c6eGDx+u7t27q6ioSJGRkZKkyMhIFRcXS5IKCgrUvXt3V2xUVJQKCgpqvOfy5cu1fPlySdKMGTMUExPjem5XjVf/T9XXHc0kzpfLqk9cVaGhoV6/lrjGs6ymHhcIOfojDgAQOHzWXDudTj311FPat2+fnn76aW3dutXtay3L8uo9k5KSlJSU5Po5Ly/PqzhvX2dHnC+XVZe4mJgYo2UQ599lNfW4QMixoeLi4uLq/H4AgMbH54OYW7ZsqV69emnt2rVq3bq1CgsLJUmFhYWKiIiQJEVHRys/P98VU1BQoKioKF+nCgAAANSJT85cFxcXKyQkRC1bttShQ4e0bt06/fGPf1RiYqLS09OVnJys9PR0DRgwQJKUmJioefPmacSIESosLFROTo4SEhJ8kSoABD2u8AQA5nzSXBcWFmr+/PmqqKiQZVkaOHCgTj/9dPXo0UNz5szRihUrFBMTo5SUFElSfHy8Bg4cqJSUFDmdTo0ePZorhQCAj3CFJwAw55PmunPnznryySdrPN6qVSs98sgjtcaMHDlSI0eObOjUAABH8XSFp2nTpkk6coWnadOm6cYbb3R7hacePXr48VMAgH9w+3MAQA0NcYUnyf1VnrjqUsPEBUKOgRIXCDk29bhAyFGiuQYA1KIhrvAkmV3liasumccFQo6BEhcIOTb1uMaUo6crPDEgDgDgFld4AoC6obkGAFRTXFysffv2SZLrCk8dO3Z0XeFJUo0rPGVkZOjw4cPKzc3lCk8AghrDQgAA1XCFJwAwR3MNAKiGKzwBgDlOLQAAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATbjONQAAXiofc5nr/7uqPB6y4APfJwOgUeLMNQAAAGATmmsAAADAJjTXAAAAgE0Ycw0AQAOqOk5bYqw20NRx5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATZjQCABAI8RESCAwceYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANgn1dwLwnfIxl7n+v+uo50IWfODbZAAAAJognzTXeXl5mj9/vvbs2SOHw6GkpCRdfPHFWrp0qb744gtFRERIkq677jr1799fkrRs2TKtWLFCTqdTo0aNUr9+/XyRKgAAAGDMJ811SEiIbrrpJnXr1k0HDhzQ/fffr759+0qSLrnkEl122WXVXr9t2zZlZGRo9uzZKiws1GOPPaa5c+fK6WQUCwAAABovn3SrkZGR6tatmySpRYsW6tixowoKCty+PjMzU4MGDVKzZs3Url07xcbGKisryxepAgAAAMZ8PuY6NzdXv//+uxISEvTLL7/os88+08qVK9WtWzfdfPPNCg8PV0FBgbp37+6KiYqKqrUZX758uZYvXy5JmjFjhmJiYlzPHT2muKqqrzuaSZwvl+WPuKpCQ0O9fm2wxAVCjoESFwg5+iMOABA4fNpcl5aWatasWbr11lt1/PHH68ILL9SVV14pSXrrrbe0aNEijR8/XpZlefV+SUlJSkpKcv2cl5fnVZy3r7MjzpfL8kVcTEyM0TKaclwg5BgocYGQY0PFxcXF1fn9AACNj8+a67KyMs2aNUvnnHOOzjzzTElSmzZtXM8PGzZMM2fOlCRFR0crPz/f9VxBQYGioqJ8lSoABDUmoQOAOZ8015Zl6cUXX1THjh01YsQI1+OFhYWKjIyUJK1Zs0bx8fGSpMTERM2bN08jRoxQYWGhcnJylJCQ4ItUASDoMQkdAMz5pLn+9ddftXLlSnXq1EmTJ0+WdOSMx9dff63NmzfL4XCobdu2Gjt2rCQpPj5eAwcOVEpKipxOp0aPHk2RBgAfiYyMdJ34qM8k9B49evgqZQBoNHzSXJ988slaunRpjccrv06szciRIzVy5MiGTAsAcAx2TkKX3E9ED5QJ3u7iGlOOVQXKpN1AiAuEHJt6XCDkKHGHRgCAG3ZPQpfMJqI31gne/lpWXeIa06TdQI8LhBybelxjytHTJHTGWgAAanA3Cd3pdMrpdGrYsGHKzs6WxCR0AKiK5hoAUI2nSeiVjp6EnpGRocOHDys3N5dJ6ACCGsNCAADVMAkdAMzRXAMAqmESemArH/O/SyUePSkyZMEHvk0GCEKcWgAAAABsQnMNAAAA2ITmGgAAALAJY65xTIzfAwAA8A5nrgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATmmsAAADAJlznGgAAcE8DwCacuQYAAABswplrNBjOggAAgGDDmWsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATbiKDRqXqjWek6jef4cYzAACgsePMNQAAAGATmmsAAADAJjTXAAAAgE0Yc40mgbHaAACgMeDMNQAAAGATzlwDAABjVb853HXUc3xziGDEmWsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANjFurtevX68NGzbYmQsAoIFQswHAN7xurqdOnapffvlFkpSWlqa5c+dq7ty5eu+99xosOQCAGWo2APiH1831f//7X/Xo0UOS9MUXX2jq1Kl6/PHH9a9//avBkgMAmKFmA4B/eH2da8uyJEk7d+6UJJ1wwgmSpH379jVAWgCA+qBmA4B/eN1cn3TSSXr55ZdVWFioAQMGSDpStFu1atVgyQEAzFCzAcA/vB4WMmHCBB1//PHq3Lmzrr76aknSjh07dPHFFzdYcgAAM9RsAPAPr89ct2rVStdff321x/r37297QgCA+qNmA4B/eN1cl5WV6auvvtLmzZtVWlpa7bmJEyfanhgAwBw1GwD8w+vm+rnnntOWLVt0+umnq3Xr1nVaSF5enubPn689e/bI4XAoKSlJF198sUpKSjRnzhzt3r1bbdu21d13363w8HBJ0rJly7RixQo5nU6NGjVK/fr1q9MyASCY1admA75QPuYy1/93VXk8ZMEHvk8GsJHXzfWPP/6o5557Ti1btqzzQkJCQnTTTTepW7duOnDggO6//3717dtXX331lfr06aPk5GSlpaUpLS1NN954o7Zt26aMjAzNnj1bhYWFeuyxxzR37lw5ndxQEgC8UZ+aDQAw53W3GhMTo8OHDxstJDIyUt26dZMktWjRQh07dlRBQYEyMzM1ZMgQSdKQIUOUmZkpScrMzNSgQYPUrFkztWvXTrGxscrKyjJaNgAEo/rUbACAOa/PXJ977rl66qmndNFFF6lNmzbVnuvdu7fXC8zNzdXvv/+uhIQEFRUVKTIyUtKRBry4uFiSVFBQoO7du7tioqKiVFBQUOO9li9fruXLl0uSZsyYoZiYGNdzu2q8+n+qvu5oJnG+XFZTj/N1jkcLDQ2t0+tNY4jz/7ICKc5EfWo2Q/kAwJzXzfWnn34qSXrzzTerPe5wOPTcc8959R6lpaWaNWuWbr31Vh1//PFuX1d584NjSUpKUlJSkuvnvLw8r+K8fZ0dcb5cVlOP88WyYmJi6rwckxji/L+sxhYXFxdX5/fzpD41m6F8AGDO6+Z6/vz59VpQWVmZZs2apXPOOUdnnnmmJKl169YqLCxUZGSkCgsLFRERIUmKjo5Wfn6+K7agoEBRUVH1Wj4ABJP61OzIyEjXt4pHD+WbNm2apCND+aZNm6Ybb7zR7VC+ytuvA0Aw8clpBcuy9OKLL6pjx44aMWKE6/HExESlp6dLktLT0113EUtMTFRGRoYOHz6s3Nxc5eTkKCEhwRepAgCq8HYoX3R0tCvG3VA+AAgGHs9c33333ZozZ44k6c9//rPb173wwgseF/Lrr79q5cqV6tSpkyZPnixJuu6665ScnKw5c+ZoxYoViomJUUpKiiQpPj5eAwcOVEpKipxOp0aPHs3XiwBwDHbV7Ep2D+WT3M+VCYQ5IZ7iAiHHQIlr6HkypnGBMkejKccFQo7SMZrrcePGKSMjQ7169dIdd9xR5zevdPLJJ2vp0qW1PvfII4/U+vjIkSM1cuRI42UCQLCxq2ZLDTeUz2SuDHNCgiuuoefJmMY1pjkawRrXmHL0NE/GY3N98skn66677tLOnTsVGxurnj17qlevXurZs6fatm1b5yQBAA3Hrpp9rKF8ycnJNYbyzZs3TyNGjFBhYSFD+QAEtWNOaJw7d6727NmjjRs3auPGjfrwww/1/PPPKyoqylW4hw0b5otcAQDHYEfNZigfAJjz6mohbdq00cCBAzVw4EBJ0r59+7R8+XJ99NFHWrVqFc01ADQi9a3ZDOUDAHNeNdeWZWnz5s3auHGjNmzYoP/85z+KjIzUwIED1bNnz4bOEQBQB9RsAPCfYzbXM2bM0O+//664uDiddNJJSkpK0oQJE9SiRQtf5AcAqANqNgD41zEHxe3YsUOhoaFq27atYmNjFRsbS5EGgEaKmg0A/nXMM9fz5s2rNjnmn//8p/bu3auTTjpJPXv21Mknn6wuXbr4IFUAwLFQswHAv+o1ofHdd99VcXGx3nrrrQZNEgDgPWo2APiP0YTGX3/9Vfv27dOJJ56ooUOHNnSOAIA6oGYDgP8cs7l+4okn9J///EdlZWVKSEhQr1699Ic//EE9evRQ8+bNfZEjAMBL1GwA8K9jNtc9e/bUyJEjdeKJJyo01KsT3QAAP6FmA4B/HbPyJicn+yANAIAdqNkA4F/cnxYAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwSagvFvL888/rhx9+UOvWrTVr1ixJ0tKlS/XFF18oIiJCknTdddepf//+kqRly5ZpxYoVcjqdGjVqlPr16+eLNAEAAIB68Ulzfd555+kPf/iD5s+fX+3xSy65RJdddlm1x7Zt26aMjAzNnj1bhYWFeuyxxzR37lw5nZxkBwAAQOPmk461V69eCg8P9+q1mZmZGjRokJo1a6Z27dopNjZWWVlZDZwhAAAAUH8+OXPtzmeffaaVK1eqW7duuvnmmxUeHq6CggJ1797d9ZqoqCgVFBTUGr98+XItX75ckjRjxgzFxMS4ntvlYblVX3c0kzhfLqupx/k6x6OFhobW6fWmMcT5f1mBFOcPDOcDADN+a64vvPBCXXnllZKkt956S4sWLdL48eNlWZbX75GUlKSkpCTXz3l5eV7Fefs6O+J8uaymHueLZcXExNR5OSYxxPl/WY0tLi4urs7v15AYzgcAZvxW+dq0aSOn0ymn06lhw4YpOztbkhQdHa38/HzX6woKChQVFeWvNAEgKDGcDwDM+O3MdWFhoSIjIyVJa9asUXx8vCQpMTFR8+bN04gRI1RYWKicnBwlJCT4K00AQBUNNZwvEIateYoLhBwDJa6hh/KZxgXKMLKmHBcIOUo+aq6feeYZbdiwQXv37tXtt9+uq6++Wj///LM2b94sh8Ohtm3bauzYsZKk+Ph4DRw4UCkpKXI6nRo9ejRfLQJAI+Cv4XwMWwuuuIYeymca15iGkQVrXGPK0dNQPp8015MmTarx2Pnnn+/29SNHjtTIkSMbMCMAQF21adPG9f9hw4Zp5syZkhjOBwBVcUoYAOCVwsJC1/+PHs6XkZGhw4cPKzc3l+F8AIKaXy/FBwBonBjOBwBmaK4BADUwnA8AzHBqAQAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm4T6OwHAn8rHXFbt511V/h+y4APfJgMAAAIeZ64BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgEy7FBxioegm/XUc9xyX8AAAIXpy5BgAAAGxCcw0AAADYhGEhAAAgIHBXXQQCzlwDAAAANqG5BgAAAGxCcw0AAADYhDHXAACgSWOsNnyJM9cAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOfXOf6+eef1w8//KDWrVtr1qxZkqSSkhLNmTNHu3fvVtu2bXX33XcrPDxckrRs2TKtWLFCTqdTo0aNUr9+/XyRJgAAAFAvPjlzfd555+nBBx+s9lhaWpr69OmjefPmqU+fPkpLS5Mkbdu2TRkZGZo9e7ZSU1P197//XRUVFb5IEwAAAKgXnzTXvXr1cp2VrpSZmakhQ4ZIkoYMGaLMzEzX44MGDVKzZs3Url07xcbGKisryxdpAgD+v+eff1633Xab7rnnHtdjJSUleuyxx3TnnXfqscceU0lJieu5ZcuW6Y477tBdd92ltWvX+iFjAGgc/Hb786KiIkVGRkqSIiMjVVxcLEkqKChQ9+7dXa+LiopSQUFBre+xfPlyLV++XJI0Y8YMxcTEuJ7bVWvEEVVfdzSTOF8uq6nHBUKO9YmrKjQ01OvXBktcIOTojzh/OO+88/SHP/xB8+fPdz1W+Y1jcnKy0tLSlJaWphtvvLHaN46FhYV67LHHNHfuXDmdTOsBEHz81ly7Y1mW169NSkpSUlKS6+e8vDyv4rx9nR1xvlxWU48LhBzrEhcTE2O0jKYcFwg5NlRcXFxcnd+vIfXq1Uu5ubnVHsvMzNS0adMkHfnGcdq0abrxxhvdfuPYo0cPP2QOAP7lt+a6devWKiwsVGRkpAoLCxURESFJio6OVn5+vut1BQUFioqK8leaAID/ryG/cQyUb5/cxQVCjoES15hyrCpQvulqynGBkKPkx+Y6MTFR6enpSk5OVnp6ugYMGOB6fN68eRoxYoQKCwuVk5OjhIQEf6UJADiGhv7GsbF+++SvZTX1uMaaY2P6pitY4xpTjp6+bfRJc/3MM89ow4YN2rt3r26//XZdffXVSk5O1pw5c7RixQrFxMQoJSVFkhQfH6+BAwcqJSVFTqdTo0ePZtweADQCfOMIAMfmk+Z60qRJtT7+yCOP1Pr4yJEjNXLkyAbMCABQV3zjCADH1ugmNAIA/I9vHAHADM01AKAGvnEEADOcWgAAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNuBQf4EPlYy5z/X/XUc+FLPjAt8kAAADbceYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE1orgEAAACb0FwDAAAANqG5BgAAAGxCcw0AAADYhOYaAAAAsAnNNQAAAGATmmsAAADAJjTXAAAAgE1orgEAAACbhPo7AQAAgMaofMxlrv/vOuq5kAUf+DYZBAzOXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbEJzDQAAANiE5hoAAACwCc01AAAAYBOaawAAAMAmNNcAAACATWiuAQAAAJvQXAMAAAA2obkGAAAAbBLq7wQAAACakvIxl7n+v+uo50IWfODbZOBznLkGAAAAbMKZayAAcBYEAIDAwJlrAAAAwCZ+P3M9YcIEhYWFyel0KiQkRDNmzFBJSYnmzJmj3bt3q23btrr77rsVHh7u71QBAAAAj/zeXEvS1KlTFRER4fo5LS1Nffr0UXJystLS0pSWlqYbb7zRjxkCACpxUgQA3GuUw0IyMzM1ZMgQSdKQIUOUmZnp54wAAFVNnTpVTz31lGbMmCHpfydF5s2bpz59+igtLc2/CQKAnzSK5vrxxx/XlClTtHz5cklSUVGRIiMjJUmRkZEqLi72Z3oAgGPgpAgAHOH3YSGPPfaYoqKiVFRUpOnTpysuLs7r2OXLl7sa8hkzZigmJsb13NFXVKiq6uuOZhLny2U19bhAyDGQ4qoKDQ31+rX+iguEHP0R1xg9/vjjkqQLLrhASUlJnBQBgP/P7811VFSUJKl169YaMGCAsrKy1Lp1axUWFioyMlKFhYXVxmNXlZSUpKSkJNfPeXl5Xi3T29fZEefLZTX1uEDIsTHHxcTEGC3Dl3GBkGNDxdXlxIK/NcRJkUD5g9VdXCDkGChxgZBjQ8VVFSh//HMCppa4OkfYqLS0VJZlqUWLFiotLdVPP/2kK6+8UomJiUpPT1dycrLS09M1YMAAf6YJAKjC1ydFGusfrP5aVlOPC4QcfRHXmP74byxxjSlHTycV/NpcFxUV6emnn5YklZeX6+yzz1a/fv104oknas6cOVqxYoViYmKUkpLizzQBAP8fJ0UAwDO/Ntft27fXU089VePxVq1a6ZFHHvFDRgAATzgpAgCe+X3MNQAgcHBSBAA8axSX4gMAAACaApprAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm4T6OwEAAABI5WMuc/1/11HPhSz4wLfJwBjNNQAAQABz15TTkPsHw0IAAAAAm3DmGmiiqp7JkDibAQCAL3DmGgAAALAJzTUAAABgE5prAAAAwCY01wAAAIBNaK4BAAAAm9BcAwAAADbhUnwAquESfgAAmKO5BmALmnIAABgWAgAAANiG5hoAAACwCcNCAAAAggxD+RoOzTUAv6pa4Hcd9RwFHgAQaBgWAgAAANiEM9cAAADwCsNJjo0z1wAAAIBNOHMNICAxVhsA0BjRXAMAAKBBBdNwEpprAEGFM94AgIZEcw0AAIBGyfSEiD9PpDChEQAAALAJZ64B4BiCaawgAKB+GnVzvXbtWr3yyiuqqKjQsGHDlJyc7O+UAABuULMBBDp3w0nqciKl0TbXFRUV+vvf/66HHnpI0dHReuCBB5SYmKgTTjjB36kBgFeC6Yw3NRsAjmi0Y66zsrIUGxur9u3bKzQ0VIMGDVJmZqa/0wIA1IKaDQBHOCzLsvydRG2++eYbrV27VrfffrskaeXKldq0aZNGjx7tes3y5cu1fPlySdKMGTP8kicAwLuaLVG3ATR9jfbMdW09v8PhqPZzUlKSZsyYccwCff/99xvl0JTjAiHHQIkLhBwDJS4QcvRHXCDwpmZL3tXtQNkugRAXCDkGSlwg5NjU4wIhR6kRN9fR0dHKz893/Zyfn6/IyEg/ZgQAcIeaDQBHNNrm+sQTT1ROTo5yc3NVVlamjIwMJSYm+jstAEAtqNkAcESjvVpISEiI/vSnP+nxxx9XRUWFhg4dqvj4eKP3SkpKIs6Py2rqcYGQY6DEBUKO/ogLBNTsxhkXCDkGSlwg5NjU4wIhR6kRT2gEAAAAAk2jHRYCAAAABBqaawAAAMAmNNcAAACATWiuAQAAAJvQXAMIKGVlZdVuWLJ+/Xp9+OGH+ve//+11/NGKi4ttyw8A8D/1rdn1We7RfFXrm2RznZeXp3379kmScnNz9c0332jr1q11fp/PPvvM7tRc/NUg1DXOZF3660Dyp2PtK42hIazL/tyYG9AHHnjAtU9+8MEH+sc//qFDhw7po48+0htvvOE2bv369br99ts1btw4TZ8+Xbm5ua7nHn/8cbdx+/fv1xtvvKFnn31Wq1atqvbcwoUL65R7UVFRnV4fTOyq296q6z5uZ11rbMdifT6br7ebXbzZBvVZL76u2aZ8sX+Z1mzT2mta62tjWrMb7XWuTaWlpelf//qXmjVrpksvvVQffvihTjrpJC1dulTnn3++RowYUWvcRx99VO1ny7KUlpamw4cPS5LbuOzsbC1evFiRkZG6/vrr9cILLygrK0txcXEaO3asunbtWmvcAw88oKlTpyo8PFwffPCB1qxZo9NOO00fffSRNm7cqOuvv77WuPXr1+u5557T4cOH1bVrV40dO1bt2rWTdGSnmTlzpm1xpuvS9LOtXbtW/fr1k3TkoHrttdeUnZ2t+Ph43XLLLWrTpk2tcZ787W9/07hx4+oc99e//lUPPvhgrc+Z7Cu+3t6m+7Pp8nbt2qV3331XUVFRSk5O1quvvqpNmzapY8eOuvHGG13vcbSKigp98cUXys/PV79+/XTyySe7nnv33Xd1xRVX1BoTHh4uScrIyNBf/vIXNW/eXMnJyZoyZYrbdblkyRKlpqYqPj5e33zzjaZPn66JEyeqR48etd66u9Lzzz+vDh066Mwzz9SXX36pb775RnfddZeaNWumTZs2uY0rKSmp9rNlWXrwwQdd67DyM8Cs1pjWXtN93PQY9vWx6Im7emj62Ux/RzRErbe7Zktm68XXNdu09prm6cuabVp7TWu9nTW7yTXXK1eu1Jw5c3Tw4EFNmDBBzz33nCIiIlRaWqrU1FS3O+jSpUt12mmnKT4+3rXyKyoqdODAAY/LW7hwoa6++mrt27dPDz/8sG655RY9/PDDWrdunRYuXOj2ryRfNwgmcabr0vSzvfnmm66Cu2jRIkVGRmrKlCn69ttv9dJLL+m+++6rNe7oA6KSZVkezzD89ttvbp/bvHmz2+dM9hVfb2/T/bk+DejgwYO1f/9+paam6rzzztOVV16pn376SS+88IKmTp1aa9xLL72kgwcPKiEhQa+88op69eqlW265RZK0Zs2aWgt1ixYttHXrVnXq1EmtWrXSoUOH1Lx5c5WXl3vMsayszHVTk7POOksdO3bU008/rRtuuEEOh8Nt3K5du3TvvfdKks444wy99957+stf/uJ2f6w0evRoxcTEVHusoKBAU6ZMkcPh0HPPPecxPpiY1BrT2mu6j5sew74+Fk3qoelnM/0dYVrrfVmzK19T1/Xi65ptWntN8/RlzTatvaa13s6a3eSaa6fTqebNmys0NFTNmzd3HRhhYWEe42bPnq3XXntNpaWluuqqq3TccccpPT1dV111lce48vJynXbaaZKO7KxnnXWWJKlPnz56/fXX3cb5ukEwiTNdl6afrars7Gw99dRTko78xZ6enu72taNHj1bbtm2rvbfD4ZBlWR6/0nnggQfUq1evWp+r/AqrNib7iq+3t+n+bLq8AwcO6MILL5R05KvMSy+9VJJ0/vnn69NPP3Ubl5WVpaefflqS9Ic//EELFy7U008/rbvuusvtehkzZoyeffZZde7cWa1bt9YDDzygnj17auvWrbr88svdLiskJER79uxxnRWLj4/XI488ohkzZmjXrl0e10lFRYWcziOj6EaOHKmoqChNnTpVpaWlbuNuuOEGrVu3TjfddJM6deokSZowYYLmz5/vNiZYmdQa09pruo+bHsO+PhZN6qHpZzP9HVFVXWq9L2u2ZLZefF2zTWuvaZ6+rNmmtde01ttZs5tcc921a1fNnTtXBw8eVO/evTV//nz169dP69evV8eOHd3GxcTE6J577lFmZqamT5+uSy65xKvlNWvWTD/++KP2798vh8OhNWvW6IwzztCGDRtcO0RtfN0gmMSZrkvTz1ZUVKSPPvpIlmXpwIEDsizLdZB7KvDt27fXI488UuMvTkn685//7DbuhBNO0NixY9WhQ4c6xZnsK77e3qb7s+nyHA6HduzYof379+vQoUPKzs7WiSeeqJ07d6qiosJtXNXxfiEhIRo3bpzeeecd/eUvf3FbPDt37qyZM2fqxx9/VE5Ojjp37qzo6GjdcsstatmypdtlXX/99dU+myRFR0dr2rRpHsc2nn766Vq/fr369u3reuy8885TmzZt9PLLL7uNu+yyyzR48GC99tprio6O1tVXX+3xl1YwM6k1prXXdB83PYZ9fSya1EPTz2b6O8K01vuyZktm68XXNdu09tbnZEPV92jImm1ae01rvZ01u8nd/ry8vFyrV6+Ww+HQWWedpU2bNunrr79WTEyMhg8f7tVf1AcPHtTSpUuVlZWlRx991ONrN2/erCVLlsjhcOiWW27R559/rvT0dEVFRWns2LHVxiMdraKiwrWzlZeXKzo6WqeeeqrHne2nn35SRESEunTpUu3x/fv369NPP9XIkSNtizt6XWZlZWnVqlVerUuTz/b2229X+3n48OGKiIjQnj17tHjxYk2cOLHWuE8//VQnn3xyjc8mSZ988okuuuiiWuO++eYbderUSXFxcTWeq/xFfSylpaV6++23vdpX7Nze+/bt02effeZ2e5vmaLp/VX4V73Q6NW7cOH300UfasmWLDhw4oHHjxmnAgAG1xs2bN0/nnnuu6yviSl988YUWLlyoN99885ifT5L27t2rVq1aefVaf/juu++0bNky5ebmasGCBf5Op9Exqdumtdd0H5fMjuGq6vK7xTRP03po8tlMf0eY1npf12yp7uvF1zXbtPaa7l/UbO80ueY6GBUVFal169b+TgOopri4WOHh4R7PIppYsmSJLr30UkVERCg7O1tz5syRw+FQeXm5Jk6c6PZrY08T4MaNG1drM3Isv/32m7p16+bVaw8dOqSdO3e6vm5E40ENRVPSULXXlGnN9sRT7bWr1tenZje5YSFZWVlasmRJtZWanZ2tDh06eJxBvmfPHr399ttyOBy65ppr9Mknn2jNmjWKi4vTqFGjFBkZWedcPG38/fv3a9myZSooKNBpp52ms88+2/XcwoULddttt9UaZzqbdcqUKTrjjDM0ePBgxcbG1vmzHM3TzOzS0lK9//77+vbbb5Wfn6/Q0FDFxsbqggsu0HnnnWe0vGM1Mfv379fatWtVUFAgSYqKiqrTGaW6LK9y22VmZqqoqEgOh0OtW7dWYmKikpOTa12m6ez4qsuqvDzSsZYlNcw28LTNK5e5du1a5eXlKSQkRB06dKj2dZ47dd12P/zwg2644QZJ0uLFizVp0iQlJCRox44dmjdvnmbMmFFrnKcJcAsWLKjzJZok6fPPP9ftt9/u1WubN2/uKtJ1acqDgWnddsfT+jWtoaY12/R3S31qdl2PKX/UC3dMjw27a7ZkVrf9UbNNa687x9p2vqrZnniqvXbV+vrU7CbXXP/97383mkE+f/589e/fXwcPHtSjjz6qs88+W/fff78yMzO1YMGCY85OrY2njW96iRnT2awlJSXat2+fHn30UbVp00aDBw/WoEGDFBUV5XZZpjOz582bpzPOOEOpqalavXq1SktLNXjwYL377rvasWOH25nnnnhal+np6XrnnXfUt29f1+dZv3693nzzTV155ZUaMmSIrcubM2eOTjnlFE2bNs1VXPfs2aOvvvpKs2fP1sMPP1wjxnR2vMmyJPNtYLrNMzIy9OGHH6pz5876+eef1aNHD23atEmLFy/WHXfcoc6dO9caZ7LtysvLVV5erpCQEB06dEgJCQmSpLi4ONdlq2pjOgHOE28b66PVpSkPBqZ12x1P69e0hprWbNPfLSY1WzI7pnxdLzwxPTbsrtmSWd32dc02rb2m286XNdsTT/tIQ9T6uu6XTa65Nl2pRUVFrrFon332mZKTkyVJF110kVasWGGUi6cNYXqJGdPZrOHh4br55pt18803a+PGjfr66681ZcoUnXDCCRo8eLCSkpJqxJjOzN69e7frL+0RI0bogQce0JVXXqnx48crJSXFqLn2tC7fe+89zZgxo8ZfzSUlJUpNTTVqrj0tLzc3V6mpqdUea9OmjZKTk/Xll18e873rMjvedFmm28B0m7/33nt6/PHHddxxx6m4uFjPPvusUlNTtWXLFi1YsEDTp093G1fXbTd8+HA98cQTSk5O1qmnnqpXX31VZ5xxhtavX+/x6z7TCXCSXBODnE6nysrKtHXrVrVr1874WtU01tXZ/cvQ0/o1raGmNdv0d4tJzZbMjilf1wtPTI+NhqzZkvd129c127T21qfW+6pmS2a1tz613p267pdNrrk2XalVh54fvXN4Gpa+ZcsWt38ZemJ6iRk7ZrP27NlTPXv21J/+9Cf99NNPysjIqLVQm87MPu644/TLL7/o5JNP1nfffec6CJxO5zEvxWdZlrKyslRQUCCHw6HIyEglJCQc8zPW9rw3y6uqtLRUO3bsUPv27T0OJ2nbtq3ef/99DRkypMaZidpm6Evms+NNliWZbwPTbW5Zlpo3by7pyGW4Ki/51blz52Nep7Wu2+6iiy5Sp06d9Pnnn7smGeXk5GjAgAEeJwuNGTPGNQEuNTVVn3/+uZ5//nnXBDh31qxZowULFsjhcGjMmDFatmyZjjvuOOXk5Oi2225TYmKix89XVlam0NDqpba4uFgREREe44KJSd02rb2mNdS0Zpv+bqnK25pdqa7HlK/rhWRe66tqyJotmdVtX9ds09pbn23nq5ptWntNa30lO2p2k2uuTVdqYmKiSktLFRYWpmuvvdb1+M6dO2vd+Srdd999ateunQYPHqyzzz5bJ5xwgld5ml5iRjpySZmUlBR99913mj59ug4ePHjM5dX2GZxOp/r161dj1m+lq666yu3BMmrUKLfLGjNmjF588UXl5OQoPj7edaAWFxdr+PDhbuN+/PFHLVy4UB06dHB93ZSfn6+dO3fqtttu06mnnlpr3OWXX64pU6aob9++io6OlnTklrzr1q2r9YL2laqOk/zll180d+5cxcbGaufOnRoz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66677rJGjRpljRo1ypo0aZK1ePFiq6SkxCj/xx9/3O1z//znP60777zTmjlzpjV+/HhrzZo1rudM7xJGbWhYwbx+G6Ie+pLptvPVNjepvw1Zs01rrylf1uyGqL2ear0ndd2/mtywkOzsbC1evFiRkZG6/vrr9cILLygrK0txcXEaN26c26+OaovLzs5Whw4dNHbsWHXt2rXOuSxdulRDhw6t9TnTme6ePp+nPE2WVxlz9tlnq3379g2eoyee1mWzZs2UlZVV7SLzlXmY3qHK0/JMZrqbznKvGvfbb79p9uzZrktJeYqr+lf9vn379PrrrysrK+uYs55NZ/E7nU7X1+SFhYWuMajh4eEev04z2XamM8g93XVs8+bNbp/74osvNHPmTIWFhSk3N1ezZ8/W7t27dfHFFxsP7/C0fwUj07rtjqf168sa6m553vxuMV2eyTFV33VS198RnpgeG3bXbMmsbvu6ZpvWXtN14suabVp7TWu9J3XdL5tcc71w4UJdffXV2rdvnx5++GHdcsstevjhh7Vu3TotWLDA7QXOPcUtXLjQbdy9995b6+OWZbnGCNWmpKRE+/bt06OPPqo2bdpo8ODBGjRokOvSPSafz1OeJsurjKk8IBo6R9N1OX78eC1cuFAHDhyo9jVVixYtNGHCBLdxpsubN2+ezjjjDKWmpmr16tUqLS3V4MGD9e6772rHjh21Tsb54YcfdMMNN0iSFi9erEmTJikhIUE7duzQvHnzNGPGjFqXVTXu9ddf9zruzTffdBXq119/XW3atNGUKVP07bff6qWXXtJ9991n22eTjoxxu++++xQXF6ft27drzJgxko5cC9bT9XNNtl1ubq5SU1OrPdamTRslJyfryy+/dLusBx54wO0vtn379rmNq6iocH0d2a5dO02bNk2zZs3S7t27PRZ40/0rGJnUbdP168sa6o/lmRxT9c2xrr8jTLedL2u2ZFa3fV2zTWuv6TrxZc02rb2mtd7Omt3kmuvy8nKddtppko78BVk5W7ZPnz56/fXXbY8rKipSampqjYuZW5bl9q8xyXymu2meJsvzdY6m67Jbt27661//qj179lQb43asa1KaLs9kprvpLPf6zI6vVJdZz6az+AcNGqS+fftq165dio2Nda3TiIgI3XXXXW6XZ7LtTGeQn3DCCRo7dqw6dOhQ47mqk5yO1qZNG23evNl19jQsLEz333+/XnjhBW3dutVtnOn+FYxMaobp+vVlDfXH8kyOKV/naLrtfFmzJbP66+uabVp7TdeJL2u2ae01rfV21uwm11w3a9ZMP/74o/bv3y+Hw6E1a9bojDPO0IYNG1y36rQzrn///iotLa31a0t3fzkdrS4z3U3zNF2eL3Osz7q0/v8M/srZyxUVFWrdurXHS1aZLs9kprvpLHfTONNZz/W5Ykt4eLh27dqln3/+2TVj/Vizzivzqcu2M51BftVVV7n9DJ4m4UycOLHGRKmQkBBNnDjR43FjR20IFiY1w3T9+rqG+qNm1/WY8nWOptvOlzVbMqu/vq7ZklntrU+t91XNNq29prXezprd5K4WsnnzZi1ZskQOh0O33HKLPv/8c6WnpysqKkrjxo3TSSedZGucKdOZ7qZ5mizP1zmaMr0jlinTme4ms9xN40xnPZt+tg0bNmjRokVq2bKlfvvtN5100knat2+fqxC6Ozthuu1MZ5Bv375dBQUF6t69e4PPPIf3fFkzfFlD/bE8k2PK1zn6mq+vTuLLmm1ae03Xia9rtim/1/o6TX8McI19FnJ9l+fLuMa0Lk3viNUQAmFdNkTc5MmTXXf42rVrl/Xkk09almVZP/74o/XYY4+5jTPZdqYzyBti5jkani/rb2M6pkzj7K6HjanWN4RA/71pWntNl+fLmm2qMdR6777vaSKWLl3q0zhTvs7TJK4xrUvTO2I1hEBYlw0RV1FRoYiICElHboaQl5cnSerbt68KCgrcxplsu8oZ5Pfdd5+mTp2qd999Vx9//LEkz1+fmsbBv3xZfxvTMWUaZ3c9bEy1viEE+u9N09prujxf1mxTjaHWN7kx176ehWzK13maxAXKurTj7pp1EQjr0tdx3bp10wsvvKA+ffooMzPTNT7t4MGDHm8NbbLtTGeQm8ah4fmyZgTKMWUaZ3JMBUqtN9WUf2+a1l5f7l++rr2NodY3ueba17OQAyVPk7hAWZemtzo1FQjr0tdxY8eO1RdffKFff/1Vffr0cRVZh8NR4xJMVZlsO9MZ5KZxaHi+rBmBckyZxpkcU4FS60015d+bprXXl/uXr2tvY6j1Ta659vUsZFO+ztMkLlDWpXTk0jsN0UjXJhDWpa/jQkNDNXz48BqPN2/eXG3btnUbJ9V925nOIDeNQ8PzZc0IlGOqPuukrsdUINV6E03596Zp7fXl/uXr2tsYan2Tu1oIgs/+/fu1bNkyZWZmqri4WJLUunVrJSYmKjk5ucZf5rBf5d2+1qxZo7y8PK/v9sW2A+zFMRVcTGuvKfYv79BcI+A9/vjjOuWUU3TeeefVuED9unXrGtXXk03Vk08+qTPOOEN9+vSpcbevqKgotzckYNsB9uKYCi6mtdcU+5d3gupqIWiacnNzlZycXO0OUZW3Vq2cOY2GVXm3r+joaI0YMULff/+9OnTooPHjx2vNmjVu49h2gL04poKLae01xf7lHZprBLzKW6vu2bPH9diePXuUlpbm8daqsE/l3b4k1eluX2w7wF4cU8HFtPaaYv/yDsNCEPBKSkqUlpam7777rsatVZOTk13FBg2n8m5fO3bsUKdOnby+2xfbDrAXx1RwMa29pti/vENzDQAAANikyV2KD8Fp+/btyszMVEFBgRwOhyIjI5WYmOizy/NB2rlzp9asWaP8/HyFhIQoNjZWZ599to4//niPcWw7wF4cU8HFtPaaYv86NsZcI+ClpaXpmWeekSQlJCToxBNPlCTNnTtXaWlp/kssiHz88cdasGCBDh8+rOzsbB06dEj5+flKTU3Vzz//7DaObQfYi2MquJjWXlPsX16ygAB35513WocPH67x+OHDh6077rjDDxkFn5SUFKu8vNyyLMsqLS21pk6dalmWZe3evduaPHmy2zi2HWAvjqngYlp7TbF/eYcz1wh4DodDhYWFNR4vLCyUw+HwQ0bBqby8XJJ0+PBhHThwQJIUExPjerw2bDvAXhxTwcek9ppi//IOY64R8G699Vb95S9/UYcOHRQdHS1JysvL086dOzV69Gg/Zxcchg0bpgceeEDdu3fXxo0b9cc//lGSVFxc7HH2ONsOsBfHVHAxrb2m2L+8w9VC0CRUVFQoKytLBQUFkqSoqCglJCTI6eTLGV/573//q+3btys+Pl4dO3b0Oo5tB9iLYyq4mNZeU+xfx0ZzjSZjz5491WYvV72DFHyjuLjYNWO9Xbt2CgsL8yqObQfYi2MquJjWXlPsX54xLAQBb/PmzVqwYIH279+vqKgoSVJ+fr5atmyp0aNHq1u3bn7OsOnbtm2bXnnlFeXm5iovL09du3ZVUVGRevXqpVGjRrm9JBTbDrAXx1RwMa29pti/vOTf+ZRA/d17773Wf/7znxqP//rrr9a9997rh4yCz4MPPmht377dsizL2rRpk/Xss89almVZ//rXv6ynn37abRzbDrAXx1RwMa29pti/vMMAGQS8gwcPqnv37jUe79Gjh0pLS/2QUfA5dOiQ4uLiJB259ul///tfSVJSUpK2bdvmNo5tB9iLYyq4mNZeU+xf3mFYCAJev3799MQTT2jIkCGu2cv5+flKT09Xv379/JtckGjfvr3eeecd9enTR99++606d+4sSSorK1NFRYXbOLYdYC+OqeBiWntNsX95hwmNaBL+/e9/u27HKh2ZvZyYmKj+/fv7ObPgsG/fPi1btkzbtm1T586dlZycrBYtWmj//v3atm2bevTo4TaWbQfYi2MqeNSn9ppi/zo2mmsAAADAJgwLQcArLy/XihUrXH9JV14aKDExUeeff75CQ9nNG1pFRYXS09P17bffKj8/X06nUx06dNAFF1ygU045xW0c2w6wF8dUcDGtvabYv7zDmWsEvGeeeUYtW7asdQxYSUmJ7r77bj9n2PQ9//zziomJUd++ffXNN9+oRYsW6tmzp95//30lJibqoosuqjWObQfYi2MquJjWXlPsX97hTwwEvN9//11z586t9lh0dLR69Oihu+66y09ZBZfffvtN48ePlySdfPLJSk1N1TXXXKOePXvqvvvuc1vg2XaAvTimgotp7TXF/uUdLsWHgBceHq7Vq1dXmxldUVGhjIwMtWzZ0o+ZBY+QkBDt3LlT0pFiX/nVYLNmzTzGse0Ae3FMBRfT2muK/cs7DAtBwMvNzdWSJUu0fv16hYeHy7Is7d+/X6eccopuuOEGtWvXzt8pNnnr16/X/Pnz1bx5c5WVlWnSpEnq3r27iouL9cEHH+jGG2+sNY5tB9iLYyq4mNZeU+xf3qG5RpOyd+9eWZaliIgIf6cSdCzL0t69e43XPdsOsBfHVHCob+01xf7lHmOu0STs379fa9eurXbdzVNPPZWvqXzo4MGD2rBhg/Ly8hQSEqIOHTqob9++cjo9jz5j2wH24pgKLqa11xT717Fx5hoBLz09Xe+884769u2rqKgoSUdmL69bt05XXnmlhgwZ4ucMm76MjAx9+OGH6ty5s37++Wf16NFDlmVp69atuuOOO1x3DTsa2w6wF8dUcDGtvabYv7xkAQHuzjvvtEpKSmo8vnfvXuvOO+/0Q0bB55577rFKS0st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h8rLyzVx4kT16tXL3yk2eaWlpXr//ff17bffKj8/X6GhoYqNjdUFF1yg8847z23c2rVr1a9fP0lHxgu+9tprys7OVnx8vG655Ra1adPGJ/kDTQX1MLiY1l5T1GzvcJ1rBLwffvhBERERkqTFixdr0qRJevbZZ/XQQw9p0aJFfs4uOMybN0/t27dXamqqrrrqKl100UWaOHGi1q9frzfeeMNt3Jtvvun6/6JFixQZGakpU6boxBNP1EsvveSL1IEmhXoYXExrrylqtndorhHwysvLXRM3Dh06pISEBElSXFycDh8+7M/Ugsbu3bt13nnnKTo6WiNGjND333+vDh06aPz48VqzZo1X75Gdna1rr71Wbdu21YgRI2qMJQRwbNTD4GJH7TVFzXaPYSEIeMOHD9cTTzyh5ORknXrqqXr11Vd1xhlnaP369erSpYu/0wsKxx13nH755RedfPLJ+u677xQeHi5JcjqdNWaXV1VUVKSPPvpIlmXpwIEDsixLDodDUs1Z6QCOjXoYXExrrylqtndorhHwLrroInXq1Emff/65cnJyVF5erpycHA0YMEBXXHGFv9MLCmPGjNGLL76onJwcxcfH689//rMkqbi4WMOHD3cbN2zYMB04cECSNGTIEO3du1cRERHas2cPjQBggHoYXExrrylqtne4WgiatC+//FJDhw71dxpBzXQbsO0Ae3FMBRdfb2/2r/9hzDWatKVLl/o7haBnug3YdoC9OKaCi6+3N/vX/zAsBAHv3nvvrfVxy7JUVFTk42yCk+k2YNsB9uKYCi6+3t7sX96huUbAKyoqUmpqqlq2bFntccuy9PDDD/spq+Biug3YdoC9OKaCi6+3N/uXd2iuEfD69++v0tLSWidTcMME3zDdBmw7wF4cU8HF19ub/cs7TGgEAAAAbMKERgAAAMAmNNcAAACATWiuATfee+89vfjii169dunSpZo3b14DZwQAcIeajcaC5hpN2oQJE/TTTz9Ve+yrr77yalbzyJEjdfvttzdYHgCA6qjZaAporgEAAACbcCk+BLWCggK9/PLL2rhxo8LCwnTJJZfo4osvlnTka8OdO3fqzjvvlCSlp6frrbfeUmlpqS6++GJ9+eWXGjdunPr27StJKisr03PPPac1a9YoJiZGEyZM0Iknnqhnn31WeXl5mjlzppxOp6688kr98Y9/9NtnBoBARc1GIODMNYJWRUWFZs6cqS5duuhvf/ubHnnkEX388cdau3Ztjddu27ZNCxcu1J133qmXXnpJ+/fvV0FBQbXXfP/99xo0aJBeffVVJSYm6uWXX5Yk3XHHHYqJidGUKVP0+uuvU6QBwAA1G4GCM9do8p566imFhIS4fi4rK1PXrl2VnZ2t4uJiXXnllZKk9u3ba9iwYcrIyFC/fv2qvcc333yj008/XSeffLIk6ZprrtEnn3xS7TUnn3yy+vfvL0k699xz9c9//rMBPxUANE3UbAQ6mms0eZMnT3Z9DSgdmRzzxRdfaPfu3SosLNStt97qeq6iokI9e/as8R4FBQWKiYlx/XzcccepVatW1V7TunVr1/+bN2+uw4cPq7y8vNovCQCAZ9RsBDqaawStmJgYtWvXzqvLMUVGRmrHjh2unw8dOqS9e/c2ZHoAgCqo2QgUjLlG0EpISFCLFi2UlpamQ4cOqaKiQlu3blVWVlaN15511ln6/vvv9euvv6qsrExLly6t07LatGmj3Nxcu1IHgKBDzUagoLlG0HI6nZoyZYo2b96sCRMmaPTo0frb3/6m/fv313htfHy8/vSnP+mZZ57R2LFjFRYWpoiICDVr1syrZSUnJ+vdd9/Vrbfeqg8++MDujwIATR41G4HCYVmW5e8kgEBTWlqqW2+9VfPmzVO7du38nQ4AwANqNnyJM9eAl7777jsdPHhQpaWlWrRokTp16qS2bdv6Oy0AQC2o2fAXJjQCXvruu+/03HPPybIsnXjiiZo0aZIcDoe/0wIA1IKaDX9hWAgAAABgE4aFAAAAADahuQYAAABsQnMNAAAA2ITmGgAAALAJzTUAAABgk/8H6nEquENp4x4AAAAASUVORK5CYII=
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Once again, since most of the data is between the 170-190cm range, that's where most of the wins are contained as well. What's interesting is that 167cm snuck inbetween a few of the values in that range, namely 170, 188, and 190.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>The bar graph below displays the total number of fighters by division in UFC history.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[33]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Number of Fighters by Division in UFC History&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Weight Division&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAz0AAAH0CAYAAAAXEEMgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAACARklEQVR4nOzdeVyVZf7/8ddhBxEFERURwQU45pZbho6ooTVm27SZ5RYtjqaWilnmuA1upGWKWY6pOepko2VamZFYmZom44qAa4JIiKgogmz37w9/nm8kIiqLnN7Px4PHnHMv1/257+vY8Oa67+uYDMMwEBERERERsVI2lV2AiIiIiIhIeVLoERERERERq6bQIyIiIiIiVk2hR0RERERErJpCj4iIiIiIWDWFHhERERERsWoKPSIidxiTycS///3vyi6jiPz8fJ5//nlq1aqFyWRi8+bNN92Gn58f//znP29qn65du/LCCy/c9LEq2p3QZwMHDiQ0NPSm9rmZujdv3ozJZCI5OflWyrspt3Iu1qCqfN5FqiKFHhGR/2/gwIGYTCZee+21a9bdCb/UVqbVq1ezYsUK1q1bx6lTpwgODi52O5PJdM2Pk5MTADt37iz22t6uF154ga5du5Z5u3eCrl27Wq6jvb09Xl5ehISEMHv2bLKzs4tsO2fOHD799NObav/UqVM88cQTpdo2ODiYU6dO4e3tfVPHuBW3ci7FuV7Q3rJlCyaTiePHjwP/F+j++PPAAw8U2e+rr77igQceoFatWjg7OxMYGMjgwYNJTEy8bg0TJ06kSZMmpapvzZo1zJ49u1TnlpycfMt/gBD5M1LoERH5HWdnZ6Kiokr8Jaaqys3NveV9Dx06RP369QkODqZu3bo4ODhcd9t58+Zx6tQpy8+vv/4KQO3atalWrdot11ARbucalZe+ffty6tQpjh8/zrfffssTTzzBrFmzaNOmDWlpaZbtatSogbu7+021XbduXUsovREHBwfq1q2LjU35/+pwK+dSFmJjY4t8dleuXGlZN3nyZB566CEaNWrEmjVriI+P56OPPsLBwYG33nqrTI7v4eGBm5tbmbR1M+7Ez71IWVPoERH5neDgYNq2bUt4eHiJ2xU38hMaGsrAgQMt7/38/Bg/fjx///vfqVGjBl5eXsybN4/Lly8zbNgw3N3dqV+/PvPmzbum/TNnzvD4449TrVo1vL29r/nr78WLFxkxYgT169fHxcWFu+++mzVr1ljWHz9+HJPJxPLly+nVqxfVqlXjzTffLPZcDMPg7bffplGjRjg4ONC4cWPeffddy/quXbsyfvx4jh49islkws/Pr8RrU6NGDerWrWv5qVOnjuV6/P6v2mfOnOHJJ5+kWrVq1KlTh/HjxzNgwIBib2uaMmUKdevWxcPDg4EDB5KVlQVc+Sv6okWL+P777y1/nV+yZMltXaO8vDxGjhyJj48Pjo6O1KtXjz59+pR4zlfP53p9NmDAAHr27HnNPt26dSvymSmOs7MzdevWpX79+rRq1Yphw4bx888/k5qaytixYy3b/f6WsG+//RZbW1uSkpKKtPXJJ5/g5OTEuXPngGs/x//6178wm804OTlRq1YtunTpYrmdrbjb27Zv306XLl1wdnbG3d2dvn37FgliV0c51q5dS1BQENWqVaNbt24cOXKkxHP+4+1tV99/+OGHNGzYEDc3Nx555BFOnz5dYjs3q3bt2kU+u1eD165du5gwYQIRERHMnz+fkJAQGjZsSKdOnXjvvff44IMPyuT4f7y9bcuWLXTq1Inq1atTvXp1WrVqxTfffANAgwYNgCufoT/+u1y6dCnNmjXD0dERHx8f3nrrLfLz84scJywsjPHjx1OvXj3q16/PhAkTCAwMvKamQYMGWe1Iqvy5KPSIiPzBO++8w7p164iJibnttubOnUvTpk3ZtWsXw4cPZ/jw4Tz22GP4+/uzc+dOXnnlFYYPH05cXFyR/SZNmkTXrl353//+x+uvv86YMWMsv7AbhsFDDz3Enj17+OSTT9i/fz9///vf6dOnD999912Rdl5//XX69u3Lvn37GDp0aLE1zp8/n/HjxzN27FgOHDhAeHg4Y8eOZdGiRcCVW25GjRqFn58fp06dYufOnbd9XeDKL1N79uxh/fr1bNq0ieTkZD7//PNrtvvvf/9LRkYGmzdvZsWKFXz++efMnDkTgNGjR9O3b1/uvfdey1/nn3766du6RnPnzmXVqlX8+9//5tChQ3zxxRd07NjxhudTUp8NHjyY6Ohojh07Ztn+yJEjfP/997z44os3fe18fHx49tlnWb16NYWFhdesv++++6hXr941wXzZsmU88sgj1KxZ85p9du3axeDBg3njjTdISEhg8+bN9O/f/7o1pKam0rNnT3x8fNix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2IG7uzteXl6laue5557j0qVL151O+urISHlo3rw5I0eO5OuvvyYsLIwPP/wQ+L8Q+MfP/l133VXsv2dnZ2caNWpU4rFsbGx44YUXLJ/R4ODgO6YvRW6XRnpERK6jevXqTJkyhREjRlyzLjQ0lAULFtClSxeqV69OREREmU77un79eubNm8f999/Phg0b+OSTT/jPf/4DQPfu3QkNDeVvf/sbM2bMoFWrVpw9e5atW7fi5OR007dMvfHGG4waNYqmTZvStWtXNm3axPvvv09UVFSZnc8fNW3alIceeoihQ4fywQcfULt2bWbNmkVmZuZN30bk7+/Pp59+yoEDB6hTpw7Vq1e/rWsUGRmJt7c3rVu3xsXFhZUrV2Jra0tAQECJdZTUZ1e9/PLLltvg/jihwvVkZ2eTmppKYWEh6enp/PDDD0ybNo369eszbdq0Evd95plnGDVqFAMHDqRly5ZFbrf8o7Vr13L06FG6dOlC7dq12bVrF0lJSZaA+kevvPIKc+bMYeDAgbz55pucO3eOIUOG0LlzZ/7yl7+U6tzK24gRI+jRowfh4eGW27Q2b97MO++8w+jRo0s9/Xa7du34xz/+wZtvvklSUhJPP/00DRs2JCUlhVWrVnHy5ElWrVpVprUfPnyYhQsX8tBDD9GgQQNSUlL48ccfadOmDQCenp64urqyceNG7rrrLhwdHXF3d+eNN97goYceYvr06fztb39j9+7dTJw4kVGjRpVqlCosLIxJkyaRmJhoCVgi1kAjPSIiJQgLC6Np06bXLH/77bdp3ry55X77Ll260L59+zI77j/+8Q+io6Np1aoVU6dOZdq0aZYvkTSZTHzxxRf87W9/Y+TIkQQFBfHggw/y5Zdf0rhx45s+1t///ncmT57M1KlTadasGTNmzGD69OmEhYWV2fkUZ/HixTRv3py//vWvdO3alfr169OjR4+bfn4gLCyM9u3bExwcTO3atVm5cuVtXSM3Nzdmz57NvffeS4sWLfjss89YvXp1sdP5/l5JfXZV+/btadGiBY0bNyYkJKRU57dixQrq1atHw4YN6d69O59++imjRo3il19+wdPTs8R9PT09efDBB9m9e/cNb6Vzd3dn3bp1PPDAAwQEBDBmzBjeeuut6860VqdOHTZu3EhycjLt27end+/eNG/enNWrV5fqvCpC9+7d2bx5M7GxsYSG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Tr//7zJrS0REbk7eI90ruwSr4enpSXp6emWXITdJ/Vb13Eqf/f6P+n+kkR4BYNWqVezbt4+8vDxatmxJ+/btK7skEREREZEyodAjAPTv37+ySxARERERKReayEBERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqafY2qRD6joiqRd9nUDWp36oe9ZmISMXQSI+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqKfSIiIiIiIhV0+xtUiGMzyIquwS5CacruwC5Jeq3qkd9VvWoz6om9VvFMD02rrJLuC6N9IiIiIiIiFVT6BEREREREaum0CMiIiIiIlZNoUdERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqKfSUs379+l2zbOPGjXz//fcl7rd582YWLVpU7Lo1a9aUSW0LFiwgOTm5xG2ioqLYvn37NcvT0tLYsmVLmdQhIiIiIlKeFHoqQc+ePQkJCbnl/T/77LMyqWPw4MH4+Pjc0r6nT59W6BERERGRKsGusgv4M1q1ahVOTk48/PDDHD58mAULFuDo6EhQUBC7d+9m1qxZAJw9e5aIiAh+++03OnTowHPPPcfy5cvJzc0lPDycBg0a0LBhQ+zt7enVqxdLlizh119/ZcKECezbt4+YmBiGDx/Onj17WLVqFfn5+dSpU4chQ4bg5OTExIkT6devH40bN2bTpk2sXbsWd3d36tati729PWFhYQDExcWxfv16zp07x3PPPUfHjh1ZsWIFycnJhIeHExISQu/evSvzkoqIiIiIXJdGeirZ+++/z4svvkhERAQ2NkW74/jx47z22mu8/fbbbN26lfT0dJ599lkcHByIjIxk+PDhmM1m4uPjATh69Cg5OTnk5+cTHx+P2WwmMzOTNWvWMH78eGbMmEGjRo1Yv359keNkZGSwevVqIiIieOutt0hJSSmy/ty5c0yePJmxY8eyfPlyAPr27YvZbCYyMlKBR0RERETuaBrpqURZWVlkZ2cTGBgIQOfOnYmNjbWsb968OS4uLgD4+PiQnp6Op6dnkTYaNWrE0aNHyc7Oxt7eHn9/f44ePUp8fDyDBg3i0KFDJCcnM378eADy8/MJCAgo0sbhw4cxm824uroC0LFjR06dOmVZ3759e2xsbPDx8eH8+fOlOrfo6Giio6MBmD59+s1cFhERERGRMqXQU4kMwyhxvb29veW1jY0NBQUF12xjZ2dH7dq1iYmJISAggIYNG7J//35SU1OpX78+qamptGjRgldfffWW6/x9HTeq+arQ0FBCQ0Nv+ZgiIiIiImVFt7dVIldXV5ydnUlMTATgp59+KtV+dnZ25OfnW96bzWbWrVuH2WwmKCiIb7/9Fj8/P0wmEwEBASQkJJCamgrA5cuXr7l9rUmTJhw8eJCLFy9SUFDAzz//fMManJ2dyc7OLu2pioiIiIhUGo30lLPc3FwGDx5sef/H518GDx7MBx98gKOjI3fddZfldraS3HfffYSHh+Pv7295ruezzz4jICAAJycnHBwcMJvNALi5uTF06FDmzJlDXl4eAH369MHb29vSnoeHB4899hjjxo3D3d0dHx+fG9bh6+uLra2tJjIQERERkTueySjt/UpSLnJycnBycgLg888/5+zZswwaNKjS6igoKCAyMpLu3bvToUOHMmv/ZNTQMmtLRERERO48psfGlVlbnp6epKen39Q+v/+j/h9ppKeSxcbG8tlnn1FYWIinpydDh1ZOOFi1ahX79u0jLy+Pli1b0r59+0qpQ0RERESkrCn0VLLg4GCCg4Mruwz69+9f2SWIiIiIiJQLTWQgIiIiIiJWTaFHRERERESsmkKPiIiIiIhYNYUeERERERGxago9IiIiIiJi1TR7m1SIspy3XcrfrcyNL5VP/Vb1qM+qHvVZ1aR+E430iIiIiIiIVVPoERERERERq6bQIyIiIiIiVk2hR0RERERErJpCj4iIiIiIWDXN3iYV4siXL1V2CX9ajR/8sLJLEBEREalUGukRERERERGrptAjIiIiIiJWTaFHRERERESsmkKPiIiIiIhYNYUeERERERGxago9IiIiIiJi1RR6RERERETEqin0iIiIiIiIVSv30LNkyRK+/PJLy/uIiAgWLFhgef/xxx+zfv368i7jGlFRURw4cKBU227evJmwsDDCw8MZOXIks2bN4vLly7d03KysLL755ptb2rcsZWRkMGvWrBtu169fv2KX79ixg+Tk5LIuS0RERESkzJV76AkMDCQhIQGAwsJCMjMzSUpKsqxPSEggMDCwvMu4bcHBwURGRjJ79mzs7OzYunXrLbWTlZXFxo0by7i6m+fh4cGoUaNuef+d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mMDQoUOZM2cOeXl5APTp0wdvb+9rju3l5QVAs2bNgCu32p05cwZXV1cABg0axKJFixg9ejQFBQWYzWZeeumlIm0MHDiQuXPnsm7dOtq0aYOLi8sNz/mP9YqIiIiI3KlMhmEYlV2EVK7Lly/j4OCAyWTip59+4qeffmLMmDFleowZ/7m/TNsTERERkcrXr8vicmnX09OT9PT0m9qnuAGCqzTSIxw9epSPPvoIwzCoVq0af//73yu7JBERERGRMqPQI5jNZiIjIyu7DBERERGRcqGJDERERERExKop9IiIiIiIiFVT6BEREREREaum0CMiIiIiIlZNoUdERERERKyaZm+TClFec7hL+biVufGl8qnfqh71WdWjPqua1G+ikR4REREREbFqCj0iIiIiImLVFHpERERERMSqKfSIiIiIiIhVU+gRERERERGrptnbpEKE7XyjsksQEakwi9pPq+wSRETkdzTSIyIiIiIiVk2hR0RERERErJpCj4iIiIiIWDWFHhERERERsWoKPSIiIiIiYtUUekRERERExKop9IiIiIiIiFVT6BEREREREaum0FMG+vXrV+T95s2bWbRoUSVVUzpHjhzho48+KnGbtLQ0Ro0aVey6zZs3k5GRUR6liYiIiIiUKbvKLkAqR+PGjWncuPEt779582YaNGiAh4dHGVYlIiIiIlL2NNJTzjIzM3n77bd54403eOONN4iPjwfg8OHDvPXWW4wZM4a33nqLlJQUAN58802SkpIs+0+cOJGjR48yfPhwMjMzASgsLGTYsGGcO3eOV155BcMwyMrK4umnnyYuLg6Af/zjH6SmppKTk8P8+fN54403GDNmDDt37gTgwIEDTJ8+3VLjlClTeP311/nwww8ZMmRIkWMtWLCAkSNH8s9//pPc3Fy2b9/OkSNHeO+99wgPDyc3N7diLqaIiIiIyC3QSE8ZyM3NJTw83PL+4sWLtGvXDoDFixfTu3dvgoKCSE9PJyIignfeeQdvb28mTZqEra0te/fuZcWKFYwePZrg4GC2bdtGgwYNOHv2LGfPnqVRo0b85S9/4ccff+TBBx9k3759NGzYkJo1a1KvXj2Sk5NJS0ujUaNGxMfH07RpU86cOUPdunVZsWIFzZs3Z8iQIWRlZfHmm2/SokWLIvV/+umnNG/enMcee4zdu3cTHR1tWXfq1ClGjBjB4MGD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Pz8/6tevD0D16tUJDAxk1KhRtG7dmn79+nH8+HHGjh2LnZ0dd999N3379i11DYGBgfz22288+uijANja2lKjRg08PT2xsbHBzc2NoUOHMmfOHPLy8oAr37v0+9Dj4eHBY489xrhx43B3d8fHxwcXF5cSj+vr64utra0mMhARERGRO57JKMX9ShMnTiQxMZEaNWpQq1Ytzpw5w/nz52ncuLHlWZMxY8aU+J0ycmfLycnBycmJgoICIiMj6d69u+WZorLQ+7/Ly6yt0vgwuFuFHs/aeHp6kp6eXtllyE1Sv1U96rOqR31WNanfqp5b6bPrfWUMlHKkx8fHhw4dOtCrVy/Lsg0bNnDy5EkmT57MmjVr+Oijj4iIiLipwuTOsWrVKvbt20deXh4tW7akffv2lV2SiIiIiEiZKFXo+emnn1i0aFGRZT179iQsLIywsDAefvhhvvjii3IpUCpG//79K7sEEREREZFyYVOajWrUqMGuXbuKLIuNjcXNzQ2AvLw87OxKlZ9EREREREQqVKmSyqBBg5g9eza+vr6WZ3pOnDjByJEjATh06BAPPPBAuRYqIiIiIiJyK0oVelq1asXcuXPZvXs3GRkZ3H333bRp04bq1atb1rdq1apcCxUREREREbkVpb4nzc3NjS5dupRnLSIiIiIiImXuuqEnIiKCcePGAfCPf/wDk8lU7HaTJk0qn8pERERERETKwHVDT0hIiOV19+7dK6QYsV763hwRERERqSzXDT2dO3e2vO7atWtF1CIiIiIiIlLmSvVMj2EYfPfdd/z0009cuHCBt99+m7i4OM6dO0dwcHB51ygiIiIiInLLSvU9PZ988gkxMTGEhoaSnp4OQK1atVi7dm25FiciIiIiInK7ShV6vv/+e15//XU6depkmdDAy8uLtLS0ci1ORERERETkdpUq9BQWFuLk5FRkWU5OzjXLRERERERE7jSlCj133303H3/8MXl5ecCVZ3w++eQT2rZtW67FiYiIiIiI3K5STWTQv39/5s2bx8CBA8nPz6d///60bNmSV155pbzrEyvx9637KruEP733g1tUdgkiIiIilaLE0LN161aaNWtGzZo1GTNmDOfPn+f06dN4enpSs2bNCipRRERERETk1pUYej755BNSU1OpW7cuZrOZZs2aYTabFXhERERERKTKKDH0zJkzh3PnznHw4EEO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xoZatWpZbuOzs7Nj1KhRLF68mEuXLlFQUECvXr1o0KCBZX8HBwfCwsKYOnUq1atXp0mTJjeso3r16gQGBjJq1Chat26tiQxERERE5I5lMm50b5PcURYtWoS/vz/du3cv03ZzcnJwcnLCMAwWLVpE3bp1y/R7jR757zdl1pbcmveDW5R6W09PT9LT08uxGikP6reqR31W9ajPqib1W9VzK3129Y/+xbGakZ4/g9dffx0nJ6ebnu66NKKjo/n+++/Jz8/H39//pkbBRERERETuZAo9VciMGTPKre3evXuX6ciOiIiIiMidwmomMhARERERESmOQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqafY2qRA38x0xIiIiIiJlSSM9IiIiIiJi1RR6RERERETEqin0iIiIiIiIVVPoERERERERq6bQIyIiIiIiVk2hR0RERERErJqmrJYK8eq2zMouQW6K+qtqUr9VlHfvdavsEkRE5CZopEdERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqKfSIiIiIiIhVU+gRERERERGrpu/pKaWnn34aX19fAGxsbHj++ecJDAy8pbY2b95My5Yt8fDwKMsSb9qCBQvo3bs3Pj4+190mKiqKtm3b0rFjxyLL09LSSExMpHPnzuVdpoiIiIjIbdFITyk5ODgQGRlJZGQkzzzzDCtWrLjltjZv3szZs2fLsLpbM3jw4BIDT0lOnz7Nli1byrgiEREREZGyp5GeW5CdnU21atUAyMnJYebMmWRlZZGfn0+fPn1o3749aWlpTJs2jcDAQBITE/Hw8GDMmDHExsZy5MgR3nvvPRwcHIiIiOCLL75g165d5ObmEhAQwEsvvYTJZGLixIn4+flx7NgxMjMzGTp0KJ9//jknTpwgODiYPn36sHbtWuzt7enVqxdLlizh119/ZcKECezbt4+YmBiGDx/Onj17WLVqFfn5+dSpU4chQ4bg5OTExIkT6devH40bN2bTpk2sXbsWd3d36tati729PWFhYQDExcWxfv16zp07x3PPPUfHjh1ZsWIFycnJhIeHExISQu/evSuzS0RERERErkuhp5Ryc3MJDw8nLy+Ps2fPMmHCBADs7e0ZPXo0Li4uZGZmMm7cONq1awfAqVOnGDFiBIMHD2b27Nls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aTsrK4vs7GwCAwMB6Ny5M7GxsZb1zZs3x8XFBQAfHx/S09Px9PQs8ZpER0cTHR0NYHm+SURERESkMij0lJF27dqxbNkyJk6cyIULF667nclkuqnlVxmGQYsWLXj11VevWWc2m9m9eze2tra0bNmSqKgoCgsL6devH4WFhVSrVo3IyMgS2y6Jvb295bWNjQ0FBQUlbg8QGhpKaGjoDbcTERERESlvur2tjHTr1o0nnngCX1/f624TGBjITz/9BMCWLVssy81mM1u3bqWwsJCzZ89y4MCBa/YNCAggISGB1NRUAC5fvkxKSopl/y+//JKAgADc3Ny4ePEiKSkpNGjQABcXF7y8vNi2bRtwJeAcP368SNuurq44OzuTmJgIYKnxRuzs7K4ZNRIRERERudNopKeM1KpVi169epW4zaBBg5gzZw5ff/0199xzj2V5hw4d2L9/P6NGjaJevXqYzeZr9nVzc2Po0KHMmTOHvLw8APr06YO3tzdNmzbl/Pnzlv18fX1xc3OzjB4NHz6chQsXsmbNGvLz8+nUqRN+fn5F2h88eDAffPABjo6O3HXXXZbb2Upy3333ER4ejr+/v57rEREREZE7lsm40b1N8qeQk5ODk5MTAJ9//jlnz55l0KBBZdb+0jWny6wtERERuT09OuZVdgkVytPTk/T09MouQ27CrfSZt7f3dddppEcAiI2N5bPPPqOwsBBPT0+GDh1a2SWJiIiIiJQJhR4BIDg4mODg4MouQ0RERESkzGkiAxERERERsWoKPSIiIiIiYtUUekRERERExKop9IiIiIiIiFVT6BEREREREaum0CMiIiIiIlZNU1ZLhfizfQlaVacvcaua1G9Vj/qs6lGfiVRNGukRERERERGrptAjIiIiIiJWTaFHRERERESsmkKPiIiIiIhYNYUeERERERGxapq9TSrEiY36qFUlJziH/vNQ9ajfqp6S+sy3Z36F1iIiYs000iMiIiIiIlZNoUdERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHpERERERMSqKfSIiIiIiIhVs8rQ069fv2uWbdy4ke+//77E/TZv3syiRYuKXbdmzZrr7jd06FAyMzMt7w8cOMD06dNLWW3lyMjIYNasWTfcrrhrCbBjxw6Sk5PLuiwRERERkTJnlaGnOD179iQkJOSW9//ss8/K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VHYRUrkKCwvJz8+3zBo3ZcoU5syZY7mlsCykRv56czU9a3/jjaTceHp6kp6eXtllyE1Sv1U96rOqR31WNanfqp5b6bOrcwMUp8JGeuTOdfnyZSZNmkRBQQGGYfDCCy+UaeAREREREalM+s1WcHZ21gxrIiIiImK1KmwiAxERERERkcqg0CMiIiIiIlZNoUdERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVNGW1VAh92aiIiIiIVBaN9IiIiIiIiFVT6BEREREREaum0CMiIiIiIlZNoUdERERERKyaQo+IiIiIiFg1zd4mFcL2v4ctrwueaFKJlYiIiIjIn41GekRERERExKop9IiIiIiIiFVT6BEREREREaum0CMiIiIiIlZNoUdERERERKyaQo+IiIiIiFg1hR4REREREbFqCj0iIiIiImLVFHrKyJkzZ5g5cybDhw9n2LBhLF68mPz8/HI7nmEYhIWFcfHiRQDOnj3LU089RXx8vGWbsLAwLly4cN023nrrrRseZ+jQoWRmZl6z/MCBAyQkJNxC5SIiIiIiFUuhpwwYhsHbb79N+/btee+995gzZw45OTmsXLmy3I5pMplo0qQJiYmJACQkJODv728JIikpKbi5uVG9evXrtvHPf/7zlo+v0CMiIiIiVYVdZRdgDfbv34+DgwPdunUDwMbGhgEDBvDKK6/w1FNPsW3bNnbs2EFeXh5paWl07tyZJ598EoAffviBr7/+mvz8fJo2bcoLL7yAjY0N/fr1o1evXsTGxuLg4EB4eDg1a9YsctzAwEASExNp06YNiYmJPPjgg/z888/AlRAUEBAAwBdffMG2bdvIy8ujQ4cOPPXUUwD069ePZcuWUVhYyEcffURcXBxeXl4YhkG3bt3o2LEjABs2bGDXrl3k5+czcuRI7O3t+fbbb7GxseHHH3/k+eefx2w2V8SlFhERERG5aRrpKQNJSUn4+/sXWebi4oKnpyepqakAHD58mOHDhxMZGcn27ds5cuQIycnJbN26lSlTphAZGWkJEQCXL1+madOmREZGYjab+e677645bmBgoGW05fDhw3To0IEzZ84AV0JPYGAge/bs4dSpU0ydOpWZM2dy9OhR4uLiirSzY8cOTp8+zdtvv83gwYMto0dXVa9enRkzZtCzZ0/WrVuHl5cXPXr04MEHH7TU90fR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1KSDBw+qoKBAw8PDevjwoXp7e833//rtjc/nk91u1+zsrCYnJ1VdXa3u7m65XC6dPXtWk5OTstvtkqTjx49r165d/8+HBoAcxz09AACkUTgcViKR0M6dO5cdDwQCmpqa0vnz59XX16d3797pzp07v338p0+fqq2tTdeuXZPL5dKtW7ckSefOnZMk9fX16caNGxQeAPgFpQcAgDSKxWIqKiqSzWZbdnx0dFT79u2TYRgqLi5Wa2urnjx58tvH93g8crvdstls2rNnjyKRSJpmDgDWRekBACCNioqKFIvF9P3792XHo9GoysvLze3y8nJFo9HfPn5JSYn52m63a2lp6T/PFQBWC0oPAABptGXLFhUUFGhsbGzZcYfDodnZWXN7bm5ODodD0s8SE4/HzbEvX75kdK4AsFrw9DYAANJo3bp12r9/v65evaq8vDzV1dXJZrMpFAppYmJCu3fvViAQkNvtliTdvn1bzc3NkqQNGzbow4cPikQiqqys1MDAwIrObRiGPn/+zCOrAeAfKD0AAKRZS0uLDMNQIBDQ5cuXtXbtWm3atEler1cbN27U4uKiTp06JUlqbGyU1+uVJFVWVqq1tVW9vb1as2aNOjo69ODBg98+b1tbm3w+n+LxuDo7O3mYAQD8jUdWAwAAALA07ukBAAAAYGmUHgAAAACWRukBAAAAYGmUHgAAAACWRukBAAAAYGmUHgAAAACWRukBAAAAYGmUHgAAAACWRukBAAAAYGl/AcWmucNiCqlWAAAAAElFTkSuQmCC
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>For anyone that is familiar with UFC weight classes, lightweight (155 lbs limit) and welterweight (170 lbs limit) being the most popular by fighter count shouldn't be surprising. To be able to be as big as possible, many fighters will cut twenty, even twenty-five pounds or more from their normal walking weight to compete. For example, the normal lightweight fighter will weigh in at 155 lbs or less (156 for non-championship bouts) on the day of the weigh-ins, but weeks or months after the fight, they will probably be walking around at 170-180 lbs. For welterweight they could be anywhere from 185-200 or even more. Obviously this isn't true for every fighter, but one could imagine that most of the gross population (in the US at least) probably weigh between that 170-200 lbs range.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Next let's look at wins by winning method.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[34]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">df_final</span><span class="p">[</span><span class="s1">&#39;winning_method&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">index</span>


<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="n">y</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;UFC Fight Outcomes&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Outcome&#39;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAwkAAAH0CAYAAAB2CGFiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA5oElEQVR4nO3deXxU5d3///dMJgsBEgiBxECAsMhm2ISCUTaJiPwiRUSQIrJZKogiShSxLfShFAS5KVRcbmTRWDFU4YZqRe8gq4BQIrKLRIIJYQsEAgaynt8ffJnbywwy0ZlMTF7Px4NHM9c55zqfk16dzjvXdc7YLMuyBAAAAAD/j93XBQAAAACoWAgJAAAAAAyEBAAAAAAGQgIAAAAAAyEBAAAAgIGQAAAAAMBASAAAlLJhwwbZbDZlZma6fUx6erpsNpu2bNnixcoAAOWBkAAAPtCzZ0898sgjpdozMzNls9m0YcMGSf/3wfvH/1q2bGkct23bNg0cOFAREREKCgpS06ZN9dBDDyk1NfW6NSxbtsxl348++qji4uJ04sQJRUVFefS6JcnhcGjZsmVu7Xvq1Ck9/vjjaty4sQICAlS3bl0NGjRIu3fvLvN533nnHdlstjIfBwBVESEBAH4FVq9erRMnTjj//fCv9UuXLlW3bt3k7++vf/zjHzp48KCSk5PVuHFjTZw48Sf79fPzM/o9ceKEZs+erYCAAEVGRspu993/TWRkZKhTp07aunWrXnvtNR05ckQfffSR/P391bVrV61du9ZntQFAZUdIAIBfgbCwMEVGRjr/hYeHS5KysrI0btw4PfLII0pOTlZ8fLxiYmLUqVMnvfjii1qzZs0N+/5hv5GRkQoJCXG53CglJUWxsbEKCgpS27ZttXHjRtlsNr3zzjtGf1lZWbr33nsVHBysJk2aKCkpybmtcePGKi4u1qhRo5wzF9fz2GOPqbCwUOvXr9c999yjhg0b6je/+Y2WL1+uO++8UyNHjtTly5clSdOnT1ezZs2M47ds2SKbzab09HRt2LBBw4cPlyTneUeOHOncd+HChWrdurUCAwNVr149DRo0yLnt4sWL+sMf/qC6desqKChInTp10qeffurcfm22591339Xdd9+t4OBgtWzZUhs3btTx48fVr18/Va9eXa1bt9bmzZuNGo8cOaL7779ftWrVUu3atdWnTx/t3bvXuT03N1ejRo1SZGSkAgMDFR0draeeeuq6vzMA8BRCAgD8iq1YsUL5+fn64x//6HJ77dq1PXKe48ePq3///urSpYtSU1M1b968635YnTJlioYPH649e/Zo8ODBGjVqlL755htJ0s6dO+Xn56e//e1vzpkLV3JycvTRRx9pwoQJCgkJKbX9ueee06lTp/S///u/btUfFxenV155RZKc550/f74kadq0aXr22Wc1fvx47d27V2vXrlX79u2dx44ePVqffPKJ3nnnHX355Ze6/fbblZCQoEOHDhnn+NOf/qRx48Zp9+7datWqlYYOHaoRI0bo97//vb788ku1atVKv/vd71RYWCjp6lKqO+64Q/Xq1dPmzZu1fft2tWjRQj179tSZM2ckSX/84x+Vmpqq1atX65tvvlFycrJatWrl1jUDwC9iAQDKXY8ePawxY8aUas/IyLAkWevXr7csy7KOHj1qSbKqVatmVa9e3fnvzTfftCzLssaNG2eFhIT8rBqWLl1qSTL6rV69uvX1119b69evtyRZGRkZlmVZ1tSpU61GjRpZRUVFzuM//vhjS5KVlJRk1Dp37lznPoWFhVb16tWt119/3dnm5+dnLV269Cdr++KLLyxJ1sqVK11uP3v2rCXJmj17tmVZljVt2jSradOmxj6bN2+2JFlHjx61LMuykpKSrB//396lS5esoKAga86cOS7P880331iSrI8++sho79ChgzVq1CjjuufNm+fcvmPHDkuS9fLLLzvbUlNTLUnW3r17nTV36dLF6LekpMRq0qSJs6/+/ftbI0aMcFkbAHiTwxfBBABQNkuXLtWtt97qfF2vXj1JkmVZv6hfPz+/UjcBN2rUSFlZWUbbgQMH1LlzZ/n5+TnbbrvtNpd9/vCv8A6HQxERETp16lSZ6rrRdXnqBuT9+/frypUr6tOnj8vtBw4ckCR1797daO/evbu2bdtmtLVr1875c2RkpCSpbdu2pdpOnz4t6eqsyq5du1SjRg2jn8uXLztnXsaPH6/7779f//nPf9S7d2/17dtXd999t0/vFQFQNRASAMAHAgMDdeHChVLt58+flyQFBQUZ7fXr1y+15l6SWrRoodzcXGVmZqpBgwY/qxZX/bry4w/m1/ugHhAQUGq/kpKSMtXUvHlz2e127du3T/fdd1+p7fv27ZN09folyW63lwoW15b1uKOsocOyrFLH+Pv7l+rPVdu130VJSYl69+7tXAb1Q6GhoZKku+++W999950++eQTbdiwQQ899JBiY2O1bt06I7ABgKfxpwgA8IGWLVtq165dKi4uNtp37Nghu92u5s2bu9XPAw88oMDAQL344osut+fk5PziWiWpdevW2rlzp1Hvj/+S7q6AgIBS1/1jYWFhuueee7Rw4ULl5uaW2v7Xv/5VERERuuuuuyRdnVk5ffq00e+PH/96Lbz8cJ/WrVsrKChIn3zyics62rRpI0natGmT0b5582bntp+rU6dO2r9/vzMA/vBf3bp1nfuFhYVp6NCheuONN/TRRx9p48aNzhkOAPAWQgIA+MCjjz6qkydPatSoUdq1a5fS0tL03nvvaerUqXr44YdVp04dt/qpX7++XnnlFS1atEgPPvig1q1bp/T0dKWmpmratGn67W9/65F6x48fr1OnTmncuHE6ePCg1q9fr+eff15S2f8KHxMTo/Xr1ysrK0vZ2dnX3W/hwoXy8/PTnXfeqbVr1yojI0M7d+7U7373O61fv17Lli1TtWrVJEm9evVSXl6e/vSnPyktLU3//Oc/tXDhwlLnlaQ1a9bozJkzunTpkmrUqKGnn35a06dP18KFC3X48GF99dVXmjlzpiSpadOmeuCBBzR+/Hh98sknOnTokCZOnKh9+/YpMTGxTNf9YxMmTFBxcbEGDBigzZs3Kz09XVu2bNHzzz+vrVu3SpKef/55rVy5Ul9//bW++eYb/eMf/1CNGjXUsGHDX3RuALgRQgIA+ECrVq20fft2nT9/Xvfee6/atm2rGTNm6KmnntIbb7xRpr4eeeQRbdy4UVeuXNHQoUPVokULDRo0SEePHtWCBQs8Um/9+vW1Zs0abd26Ve3bt9fEiROdsxc/Xhp1I3PnztWuXbsUExNj/MX8xxo1aqT//Oc/6tKli/7whz+oadOmuueee5Sfn69t27apb9++zn1btGihRYsW6b333tMtt9yiJUuW6K9//avRX+fOnTVx4kQ9+uijioiI0IQJEyRJL7zwgmbMmKEFCxbolltuUZ8+fYxZiDfffFN33323HnroIbVr106ff/65Pvzww1JfaFdWERER2rZtm8LDwzVw4EC1aNFCw4YN07Fjx3TTTTdJuvq7/fOf/6xbb71VnTp10p49e/Txxx87lyMBgLfYrF961xsAoEratGmTevTooT179ig2NtbX5QAAPIiQAABwy2uvvaZ27dopKipKBw4c0KRJk1S7dm1t377d16UBADyMpxsBANxy7NgxzZw5U6dOnVJkZKTuuusuvfTSS74uCwDgBcwkAAAAADBw4zIAAAAAAyEBAAAAgIGQAAAAAMDAjcvlICsry9cloIIJDw//yS+RQtXEuIArjAu4wriAK1FRUR7ri5kEAAAAAAZCAgAAAAADIQEAAACAgXsSyoH/6s98XQIqmAuS/H1dBCocxgVcYVzAFcZFxVT42zt9XYLHMJMAAAAAwEBIAAAAAGAgJAAAAAAwEBIAAAAAGAgJAAAAAAyEBAAAAAAGQgIAAAAAAyEBAAAAgIGQAAAAAMBASAAAAABgICQAAAAAMBASAAAAABgICQAAAAAMhAQAAAAABoevC/CU06dP66WXXtLcuXOdbStWrFBQUJD69+9v7DtkyBA1bNhQxcXF8vPzU48ePdSvXz/Z7Xbt379fs2fPVr169Zz7Dx8+XG3bttX58+e1bNkypaWlyeFwqF69ehoxYoSioqLK7ToBAAAAb6s0IaEsAgICNGfOHEnShQsXtGDBAuXl5Wnw4MGSpFatWmnKlCnGMZZlac6cOerRo4eefPJJSVJ6erouXLhASAAAAEClUuWXG4WGhmrs2LFau3atLMu67n779++Xw+FQnz59nG2NGzdWq1atyqNMAAAAoNxUyZmEH4uIiJBlWbpw4YIk6eDBg0pMTHRuf/rpp/Xdd98pJibGVyUCAAAA5abShASbzVam9h/74SyCq+VGZZGSkqKUlBRJ0qxZs352PwAAAIAvVJqQULNmTV26dMlou3TpkurVq+ecFbjrrruM5ULXnDp1Sna7XaGhoTp+/LjL/qOjo/XFF1+4VUt8fLzi4+PLeAUAAABAxVBpQkJQUJBq166tvXv3KjY2VpcuXdJXX32lfv36qWfPntc9Ljc3V4sWLVLfvn1/ctbhlltu0fLly5WSkuIMAEeOHFFBQYFat27t6csBAAAAfMZm/dTdur8ymZmZWrx4sXNGoX///urWrVup/X78CNRu3bopISHhuo9Avf/++9W1a1edO3dOy5Yt09GjR+Xv76+6detq5MiRuummm36yrjOvvePZCwUAAECFU/jbO316fk8+cbNShYSKipAAAABQ+VWmkFDlH4EKAAAAwERIAAAAAGAgJAAAAAAwEBIAAAAAGAgJAAAAAAyEBAAAAAAGQgIAAAAAAyEBAAAAgIGQAAAAAMBASAAAAABgICQAAAAAMBASAAAAABgICQAAAAAMhAQAAAAABoevC6gKCn97p69LQAUTHh6u7OxsX5eBCoZxAVcYF3CFcQFvYyYBAAAAgIGQAAAAAMBASAAAAABgICQAAAAAMBASAAAAABgICQAAAAAMhAQAAAAABkICAAAAAAMhAQAAAICBb1wuB9mrH/V1Cahg+I5MuMK4gCuMC7hS0cdF+G9f93UJ+IWYSQAAAABgICQAAAAAMBASAAAAABgICQAAAAAMhAQAAAAABkICAAAAAAMhAQAAAICBkAAAAADAQEgAAAAAYCAkAAAAADAQEgAAAAAYCAkAAAAADIQEAAAAAAZCAgAAAAADIQEAAACAweHrAn6O4cOHKykpSZKUmpqqZcuW6c9//rNsNpsWL16szMxMWZaljh07avjw4XI4/u8yn332WVmWJcuydOnSJRUUFCgsLEySlJiYqL/85S+aOXOmQkJC9O2332ru3LmaPHmyIiIitGTJEn399deSpBYtWmj06NEKDg4u/18AAAAA4EW/ypBwzd69e7V06VI9//zzqlOnjqZOnao+ffromWeeUUlJid544w0tX75cw4cPlySdPn1aYWFhevbZZyVJGzZsUFpamsaMGVOq72PHjmnu3LmaNGmSYmJiNHfuXEVHR2vChAmSpBUrVuj111/XU089VX4XDAAAAJSDX+1yo4MHD+qNN97QlClTFBkZqX379ikgIEC9evWSJNntdo0YMULr169Xfn6+JGn37t1q3779Dfs+fvy45syZo8cff1zNmjXTyZMn9e2332rQoEHOfQYNGqS0tDSdPHnSK9cHAAAA+MqvMiQUFRVp9uzZSkxMVP369SVJGRkZiomJMfYLDg5WeHi484O8uyFh9uzZGj16tFq2bClJyszMVOPGjWW3/9+vy263q3HjxsrMzCx1fEpKiqZMmaIpU6b83EsEAAAAfOZXudzIz89PLVq00GeffaZRo0Y52202W6l9LcuSzWZTUVGRzp49q4iIiBv2Hxsbq88++0zt27eX3W539uGqb1fi4+MVHx9fhisCAAAAKo5f5UyCzWbTpEmTlJaWppUrV0qSGjRooG+//dbYLy8vzxkMDh486JwZuJFr9yi8+eabkqTo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
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>This may shock a lot of people, UFC fans and non-fans alike. Many may guess unanimous decisions were the most popular winning method, but I don't think many would realize how close KO/TKO is behind it. To expand more on the winning methods, "Overturned" includes situations where fighter A won the bout, then for one reason or another, fighter B was awarded the win post-fight. One example of this could be if fighter A was found to be on performance enhancing drugs and the UFC decided to award the win to fighter B. A "CNC" or "Could Not Continue" is essentially the MMA equivalent of a boxer's coach throwing in the towel. Between rounds, a coach may be trying to save the health of their fighter and decalare a CNC. You're probably familiar with a "DQ" or "Disqualification" in which one fighter does something illegal like knee the head of a grounded opponent or strike the opponent with 12-6 elbows. Finally, "Other" will encompass things like split draws I mentioned at the beginning of the tutorial and no contests, which are fights that have neither a winner nor a loser due to extrenuating circumstances like an accidental headbutt or eye poke.</p>
<p>It would be interesting to examine this data further by weight class.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[35]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bar</span> <span class="o">=</span> <span class="n">df_final</span><span class="o">.</span><span class="n">groupby</span><span class="p">([</span><span class="s1">&#39;weight_class&#39;</span><span class="p">,</span> <span class="s1">&#39;winning_method&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">size</span><span class="p">()</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span><span class="o">.</span><span class="n">pivot</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="s1">&#39;winning_method&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="s1">&#39;weight_class&#39;</span><span class="p">,</span> <span class="n">values</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">bar</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">kind</span><span class="o">=</span><span class="s1">&#39;bar&#39;</span><span class="p">,</span><span class="n">stacked</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;UFC Fight Outcome by Division&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Weight Class&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA34AAAJjCAYAAABTFsPlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAACJ5klEQVR4nOzdd3xTZf//8XcHbSmzpUAFGSIdjDJkWlbBoreKgkxFBFmKCxVFUGQoDqwCIiKCKKigN/JVkOUA2SBDkU1btpQWSpcFutv8/uDX3I2UnfS0J6/n48GD5OQk/Vw9SZp3rutcl4vFYrEIAAAAAGBarkYXAAAAAABwLIIfAAAAAJgcwQ8AAAAATI7gBwAAAAAmR/ADAAAAAJMj+AEAAACAyRH8AADXbd26dXJxcVFMTMw13+f48eNycXHRpk2bHFgZwsLCNGTIEENrmDdvntzd3a/rPtdbt4uLi+bPn3+9pQGA0yL4AUAxdLkPwTExMXJxcdG6desk/S9M/ftfcHCwzf1+//13de/eXVWrVpWXl5duv/129evXTzt37rxsDfPmzSv0sYcNG6bQ0FDFxcWpWrVqdm23JLm7u2vevHnXtO+ZM2f03HPPqXbt2vLw8FDlypXVs2dP7dq167p/7vz58+Xi4nLd93MWEyZMsD4H3NzcVLFiRTVr1kyvvPKKTp48abNvnz59dOrUqet6/B9++EFTpky55v3j4uLUs2fP6/oZAODMCH4AYAI//vij4uLirP8K9qrNnTtX7dq1U6lSpbRgwQIdPHhQCxcuVO3atfX8889f8XHd3NxsHjcuLk4RERHy8PCQv7+/XF2N+zNy8uRJNW/eXFu2bNHMmTN1+PBhrVixQqVKlVLr1q31888/G1abWdWuXVtxcXGKiYnRtm3bNHLkSK1fv14NGjTQli1brPuVLl1aVatWva7H9vX1Vfny5a95f39/f3l5eV3XzwAAZ0bwAwAT8PX1lb+/v/Wfn5+fJCk2NlZPPfWUhgwZooULFyo8PFy33XabmjdvrrfeektLly696mMXfFx/f3+VL1++0KGeq1evVkhIiLy8vNSoUSOtX7++0OF4sbGxeuCBB+Tt7a06dero66+/tt5Wu3Zt5ebmauDAgdbepct55plnlJ2drbVr1+ree+9VzZo11bJlS3377bfq1KmTHn/8caWnp0u62FtVt25dm/tv2rRJLi4uOn78uNatW6fHHntMkqw/9/HHH7fuO2PGDNWvX1+enp6qUqWKTU/TuXPn9OSTT6py5cry8vJS8+bN9euvv1pvz++V/eabb3TPPffI29tbwcHBWr9+vU6dOqX77rtPZcqUUf369bVx40abGg8fPqwePXqoYsWK8vHx0d133629e/de7ZApLy9Po0ePlp+fn8qXL68hQ4ZYfxdz585VxYoVlZaWZnOfN954Q7fddpssFstlH9fNzU3+/v665ZZbFBQUpIcfflibNm1SgwYNNGDAAOXl5UmyHeqZmpoqb29vffPNNzaPFRcXJzc3N2tA/3cv96ZNm9SmTRuVK1dO5cqVU+PGjfXLL79Yb//3cysuLk4PP/ywKlasqNKlSyssLEx//PGH9fb85+yqVavUvn17eXt7q379+jaPCQBmRvADABP77rvvlJmZqddff73Q2318fOzyc06dOqUHH3xQrVq10s6dOzV16lSNGDGi0H1Hjx6txx57THv27FHv3r01cOBAHTp0SJK0Y8cOubm56cMPP7T2MBYmOTlZK1as0LPPPltoL9Grr76qM2fOaNWqVddUf2hoqD7++GNJsv7cadOmSZLGjx+vUaNG6emnn9bevXv1888/q0mTJtb7Dho0SL/88ovmz5+vv/76S23atFGXLl0UGRlp8zPGjh2rp556Srt27VK9evX0yCOPaMCAARo6dKj++usv1atXT3379lV2draki8NY27ZtqypVqmjjxo3aunWrgoKCFBYWprNnz16xPf/3f/+nxMREbdy4UQsWLNDSpUs1atQoSdLDDz8sFxcXLVq0yLp/Xl6e5s6dqyFDhlz3cNdSpUrppZde0uHDhwsdOly+fHl17dpVX375pc32BQsWqGrVqurcufMl98nNzbV5Pu3cuVMTJkyQt7d3oTVYLBZ169ZNkZGRWr58ubZv32597ISEBJt9X375Zb322mvavXu3mjdvrj59+iglJeW62gwAJZIFAFDsdOjQwTJ48OBLtp88edIiybJ27VqLxWKxHDt2zCLJUrp0aUuZMmWs/+bMmWOxWCyWp556ylK+fPkbqmHu3LkWSTaPW6ZMGUtUVJRl7dq1FkmWkydPWiwWi+W1116z1KpVy5KTk2O9/08//WSRZPn6669tap08ebJ1n+zsbEuZMmUsn376qXWbm5ubZe7cuVesbdu2bRZJlh9++KHQ2xMTEy2SLBERERaLxWIZP3685fbbb7fZZ+PGjRZJlmPHjlksFovl66+/tvz7z+L58+ctXl5elvfff7/Qn3Po0CGLJMuKFStstjdt2tQycOBAm3ZPnTrVevv27dstkiwffPCBddvOnTstkix79+611tyqVSubx83Ly7PUqVPH5rH+rUOHDpcci1mzZlk8PDws58+ft1gsFstzzz1nadOmjfX2n3/+2eLu7m6JjY297OMW9jvMd/DgQYsky8KFCy0Wy8Xnjpubm/X2n376yeLm5mY5deqUdVujRo0sL7/8sk3d+c/5pKQkm+d5YQo+t1avXm2RZNm/f7/19oyMDIu/v7/ljTfesFgsFutz9vvvv7fuExcXZ5Fk+fnnny/7cwDALK5vyi0AQLE0d+5cNWvWzHq9SpUqknTFYXvXws3N7ZKJUmrVqqXY2FibbQcOHFCLFi3k5uZm3XbnnXcW+pgFe8vc3d1VtWpVnTlz5rrqulq77DVJy/79+5WRkaG777670NsPHDggSWrfvr3N9vbt2+v333+32da4cWPrZX9/f0lSo0aNLtkWHx8v6WLv559//qmyZcvaPE56erq1h/RyWrZsaXMs2rRpo6ysLB05ckSNGjXSk08+qYYNG+rAgQOqX7++PvvsM91///265ZZbrvi4l5N/PC73e+/cubOqVKmiBQsWaOTIkdq9e7f27Nlz2Vk5fXx8NGTIEN1zzz3q1KmTOnTooIceekhBQUGF7r9//35VqlRJ9evXt27z9PRUq1attH//fpt9Cz7//P395ebmdt3PPwAoiRjqCQDFkKenp/75559LtucPSfv3pBbVq1dX3bp1rf/yhz8GBQUpNTX1upZd+LeCj1u3bl2VKlWq0P3+/aH/ciHAw8Pjkv3yzw27VgEBAXJ1ddW+ffsKvT1/e35QcHV1vSQs5g+pvBbXGyQtFssl9yn4e8u/rbBt+b+LvLw83XXXXdq1a5fNv6ioKE2YMOG66ymoQYMGatu2rebMmaP4+HgtXbpUTzzxxHU9ZkH5v+/bb7+90Nvd3Nz06KOP6quvvpIkffXVV2ratKlCQkIu+5ifffaZ/vzzT3Xu3Fnr169Xw4YNNWvWrMvuX9gxKuw4/Pv5J+m6n38AUBIR/ACgGAoODtaff/6p3Nxcm+3bt2+Xq6urAgICrulxevXqJU9PT7311luF3p6cnHzTtUpS/fr1tWPHDpt6/93jda08PDwuafe/+fr66t5779WMGTOUmpp6ye3vvPOOzfljVapUUXx8vM3j/vt8tPxAUHCf+vXry8vL67ITgDRo0ECStGHDBpvtGzdutN52o5o3b679+/dfEurr1q2rypUrX/G+hR0LDw8Pm2D25JNP6quvvtLs2bPl7++v//znPzdUZ3Z2tqZMmaLAwECb3rR/GzBggPbt26c//vhD3377rQYMGHDVx27YsKFGjBihn376SYMHD9bs2bML3a9BgwZKSEiw9sBKUmZmprZv337TxwEAzILgBwDF0LBhw3T69GkNHDhQf/75p44cOaL//ve/eu2119S/f39VqlTpmh6nevXq+vjjj/XZZ5/p4Ycf1m+//abjx49r586dGj9+vLp27WqXep9++mmdOXNGTz31lA4ePKi1a9dqzJgxkq6/t+y2227T2rVrFRsbe8nEHAXNmDFDbm5u6tSpk37++WedPHlSO3bsUN++fbV27VrNmzdPpUuXliR17NhRaWlpGjt2rI4cOaJFixZpxowZl/xcSVq6dKnOnj2r8+fPq2zZsnrppZc0YcIEzZgxQ9HR0dq9e7feffddSRd7uHr16qWnn35av/zyiyIjI/X8889r3759Gjly5HW1+9+effZZ5ebmqlu3btq4caOOHz+uTZs2acyYMTZLJxQmMTFRzzzzjA4ePKgVK1Zo7NixGjp0qMqUKWPdJ39m0okTJ2rw4MHXtDRHbm6uTp8+rdOnTysqKkr//e9/1bZtWx04cEBffvnlFR+jYcOGatq0qYYOHaqzZ8/qkUceuey+hw8f1qhRo7Rp0yadOHFCv//+uzZu3GgzlLOgTp06qWXLlurbt682b96sffv2qX///srIyNBTTz111XYBgDMg+AFAMVSvXj1t3bpVKSkpeuCBB9SoUSO9/fbbGjFixBWHuxVmyJAhWr9+vTIyMvTII48oKChIPXv21LFjx/TRRx/Zpd7q1atr6dKl2rJli5o0aaLnn3/e2st4vWutTZ48WX/++aduu+22K/Zs1apVS3/88YdatWqlJ598UrfffrvuvfdeZWZm6vfff7fpwQoKCtJnn32m//73v2rYsKG++OILvfPOOzaP16JFCz3//PMaNmyYqlatqmeffVbSxWD09ttv66OPPlLDhg1199132/QWzpkz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22s3IyNDhw4dUkBAgCTp8OHDysjIkCTThaF69epp0qRJNr259erVU0ZGhsqUKWNwdfbn4uKiLVu2WNu7detWgytyHGd7r3K29j722GMaO3as/P39ZbFYFB8fryFDhigjI0MdOnQwujy7KgltZXKXG7BmzRp9//33l4ShNm3aaNGiRXrssceMLtGukpOTdfjwYVksFtWtW1e+vr5Gl+RQznZ8U1NT9X//93+KjIy0/hHq1auXvL29lZCQIH9/f6NLtBtnaqszcrbX7uHDhzVz5kxr2CtdurSGDRumW2+9VTt37lRoaKjBFdqPxWLRtm3bFBkZKUkKDg5Wq1atTLvA+ZkzZzR37lwdOnRIkhQQEKDHH39cvr6+Onr0qIKDgw2uELh22dnZOnXqlKSLPZ7FYZITRynubSX43SCzh6FTp06pevXqhU47LMlUk2AUxuzHF+Y1ZcoUjRgxotCJPySZfs1CZ3ztpqWlyWKxmLLnC+blbO9Vztbeffv2qWHDhoXOyCuZa33VktRWgt91cKYwNGvWLD355JM2J6kWZMbhCM50fCVp3rx5evzxxwudflgy10LBztTW5ORk+fj4XPYcLzOuWehsr90NGzaoffv2Wr58eaG3d+nSpYgrcpyxY8dq4sSJl8xyabFY5OLioi+//NLA6uzvxx9/VNeuXQudoVaSBg0aVMQVOY6zvVc5W3u/++479e7d+7LrrJppfdWS1FaC33VwxjDkTJzt+B49elR16tS57ILBZloo2JnaWtDZs2cVFxenRo0aKSsrS7m5uSpdurTRZdmds712V61apc6dO2vRokWF3t6rV68irgj28scff6h58+Zat25dobeHhYUVaT1FxVneq/I5W3tRfBD8cEWZmZlavny5EhIS9OSTTyouLk6xsbFq1qyZ0aXBjrKyspSQkKBq1aoZXYrDOUtbV69erd9++03nz5/X9OnTFRcXp88++0zjxo0zujTghkRGRiouLk4dO3ZUamqqMjIyVKVKFaPLcqiMjAx5eXkZXYZDOdt7lbO1NyUlRd9++62Sk5P12muvKSYmRtHR0erUqZPRpdldSWiruaaxKyKZmZn6/vvvNWvWLElSXFyc/vzzT4OrcoxPPvlE7u7uio6OliRVqlRJ//3vfw2uyrGc6fhKF79hHjlypN5++21JF9ehee+99wyuyjGcqa2//PKLJk6c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Q19//bVycnI0Y8YMHT9+XAsXLrSZMMIsNmzYoPbt22v58uWF3t6lS5cirsixvvrqK0kXz9OtWLGi2rdvL4vFok2bNik9Pd3g6hzj//7v//T7779be7xmzpyp1q1bq0ePHgZX5hj5Q1qTk5P12muvKSYmRtHR0aY6raRt27Zq0qSJvvnmGz366KPW7aVLl1bZsmUNrMwxFi9erMWLF18ymZq7u7upl4+aP3++unfvLg8PD73zzjs6ceKEBgwYYMr5ItLS0nTffffZbFu5cqVB1RSO4Hcddu3apfXr1ysxMdH6h1eSvLy89MgjjxhYmf052xuyJJ06dUonT55UWlqazbkV6enppp/J1FlER0dbhwLOnDlTx48f1+rVqzVkyBCjS7O7RYsW6d1339WECRMkSbVr19bZs2eNLcpB8r94Kyz0FLdvW+1p9+7deuedd6zX7777br322mvq2rWrgVU5xubNm/Xee+9ZJ1fr1q2bRo0aZdrg98knnygsLEyLFy+WJN1yyy2aOnWqqYKft7e3vL299cILLygvL08pKSnKy8tTRkaGMjIy5OfnZ3SJdvXQQw/poYce0jfffGOzTJbZ7d69W/369dP27dvl6+urESNG6I033jBl8Fu/fv0lwW/dunWXbDMSwe86hIWFKSwsTFu3blXr1q2NLsehCr4hSxcXCs7OzjbtG7IkxcbGaufOnbpw4YLNeRZeXl568sknDawM9jJv3jyNGTNGERERki6GofyJMczGzc2tWJ9gbk+dO3eWpEKHtK5YsaKoyykyrq6u2rhxo3VWz82bN5t2CZrKlSsrOzvbGvyys7MvGZJvJufOnVNoaKiWLFki6eLr2azH9ueff9aiRYtUoUIF6xc1ZpzlMl/fvn11/vx5nT592mbIcv369Q2synFyc3MlSTt37lTbtm1N2XmwadMmbdq0SfHx8Tajw9LT04vd+rEEvxvQrFkz6wHOy8uzbi9ua3XYwx9//KGvvvpKycnJKl++vBISElS9enVNmTLF6NLsrkWLFmrRooWio6MVGBhodDlwkH9/aWHWD1M1atTQpk2blJeXp7i4OP30009O+bxevny57r//fqPLcIjhw4dr3rx5mjdvniQpKChIw4cPN7YoB3F3d9eIESPUqFEjubi4aM+ePQoODrYuCl2cJk+wB09PT507d84ahKKjo037Rc6KFSv0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
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>If you take a look at middleweight (185 lbs), light heavyweight (205 lbs), and heavyweight (260 lbs), you'll notice that KO/TKO has overtaken unanimous decision as the most common outcome. This is because with more weight comes more strength and power. While heavier fighters can usually take harder hits and stay standing compared to lighter fighters, that resilience probably doesn't scale at the same rate as the power that comes with added weight.</p>
<p>Another point of interest is the women's flyweight (125 lbs) and women's strawweight (115 lbs) divisions. The second most common method behind U-DEC is no longer KO/TKO but submissions. This is again probably due to the previous point, where those lighter women don't have as much knockout power but they can still have crisp submission technique that allows them to submit opponents without needing as much strength as the heavier divisions.</p>
<p>Other than those weight classes mentioned, every other one follows the same ranking of outcomes.</p>
<p>Finally, let's see if the distribution of outcomes has changed over the years at all.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[36]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bar</span> <span class="o">=</span> <span class="n">df_final</span><span class="o">.</span><span class="n">groupby</span><span class="p">([</span><span class="s1">&#39;fight_year&#39;</span><span class="p">,</span> <span class="s1">&#39;winning_method&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">size</span><span class="p">()</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span><span class="o">.</span><span class="n">pivot</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="s1">&#39;winning_method&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="s1">&#39;fight_year&#39;</span><span class="p">)</span>
<span class="n">bar</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">kind</span><span class="o">=</span><span class="s1">&#39;barh&#39;</span><span class="p">,</span> <span class="n">stacked</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;UFC Fight Outcomes over the Years&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Year&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>While we see some of these outcomes bounce back and forth, for the most party they've remained surprisingly stable in terms of frequency. In the earlier years of the UFC, there were definitely more knockouts happening, but that quickly change around 2006. It seems like the values taper off once 2014 comes around.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Correlations-with-Winning-Fights?">Correlations with Winning Fights?<a class="anchor-link" href="#Correlations-with-Winning-Fights?">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Hypothesis-Testing">Hypothesis Testing<a class="anchor-link" href="#Hypothesis-Testing">&#182;</a></h3>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Our null hypothesis would state that none of the fields we've covere in our dataframe would have any correlation with winning fights. The alternate hypothesis would assert that there is indeed qualities that correlate with winning fights. Let's test this.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[37]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop irrelevant columns</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">df_final</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;fights_location&#39;</span><span class="p">,</span> <span class="s1">&#39;card_name&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># Encode inputs of type object</span>
<span class="n">encoder</span> <span class="o">=</span> <span class="n">LabelEncoder</span><span class="p">()</span>
<span class="n">encoded_1</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]</span>
<span class="n">encoded_1</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">encoded_1</span><span class="p">)</span>

<span class="n">encoded_2</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;stance_f1&#39;</span><span class="p">]</span>
<span class="n">encoded_2</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">encoded_2</span><span class="p">)</span>

<span class="n">encoded_3</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;stance_f2&#39;</span><span class="p">]</span>
<span class="n">encoded_3</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">encoded_3</span><span class="p">)</span>

<span class="n">encoded_1</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">encoded_1</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">])</span>
<span class="n">encoded_2</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">encoded_2</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;stance_f1&#39;</span><span class="p">])</span>
<span class="n">encoded_3</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">encoded_3</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;stance_f2&#39;</span><span class="p">])</span>

<span class="n">df</span><span class="p">[[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]]</span> <span class="o">=</span> <span class="n">encoded_1</span><span class="p">[[</span><span class="s1">&#39;weight_class&#39;</span><span class="p">]]</span>
<span class="n">df</span><span class="p">[[</span><span class="s1">&#39;stance_f1&#39;</span><span class="p">]]</span> <span class="o">=</span> <span class="n">encoded_2</span><span class="p">[[</span><span class="s1">&#39;stance_f1&#39;</span><span class="p">]]</span>
<span class="n">df</span><span class="p">[[</span><span class="s1">&#39;stance_f2&#39;</span><span class="p">]]</span> <span class="o">=</span> <span class="n">encoded_3</span><span class="p">[[</span><span class="s1">&#39;stance_f2&#39;</span><span class="p">]]</span>

<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">df</span><span class="p">,</span><span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;winning_method&#39;</span><span class="p">],</span> <span class="n">prefix</span><span class="o">=</span><span class="s1">&#39;winning_method&#39;</span><span class="p">)],</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;winning_method&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">display</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>

<span class="n">encode</span> <span class="o">=</span> <span class="n">df</span><span class="p">[[</span><span class="s1">&#39;f1&#39;</span><span class="p">,</span> <span class="s1">&#39;f2&#39;</span><span class="p">,</span> <span class="s1">&#39;weight_class&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">encoder</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">)</span>

<span class="n">df</span><span class="p">[[</span><span class="s1">&#39;f1&#39;</span><span class="p">,</span> <span class="s1">&#39;f2&#39;</span><span class="p">,</span> <span class="s1">&#39;weight_class&#39;</span><span class="p">]]</span> <span class="o">=</span> <span class="n">encode</span><span class="p">[[</span><span class="s1">&#39;f1&#39;</span><span class="p">,</span> <span class="s1">&#39;f2&#39;</span><span class="p">,</span> <span class="s1">&#39;weight_class&#39;</span><span class="p">]]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1</th>
      <th>f1_sig_strike_per</th>
      <th>f1_sig_strike_total</th>
      <th>f1_td_attempt</th>
      <th>f1_td_succeed</th>
      <th>f2</th>
      <th>f2_sig_strike_per</th>
      <th>f2_sig_strike_total</th>
      <th>f2_td_attempt</th>
      <th>f2_td_succeed</th>
      <th>...</th>
      <th>f2_age_when_fight</th>
      <th>winning_method_CNC</th>
      <th>winning_method_DQ</th>
      <th>winning_method_KO/TKO</th>
      <th>winning_method_M-DEC</th>
      <th>winning_method_Other</th>
      <th>winning_method_Overturned</th>
      <th>winning_method_S-DEC</th>
      <th>winning_method_SUB</th>
      <th>winning_method_U-DEC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Justin Jaynes</td>
      <td>0.28</td>
      <td>182.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Charles Rosa</td>
      <td>0.47</td>
      <td>92.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Damir Hadzovic</td>
      <td>0.47</td>
      <td>219.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Yancy Medeiros</td>
      <td>0.51</td>
      <td>237.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Damir Ismagulov</td>
      <td>0.47</td>
      <td>63.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Rafael Alves</td>
      <td>0.44</td>
      <td>126.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Julija Stoliarenko</td>
      <td>0.52</td>
      <td>91.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>Julia Avila</td>
      <td>0.42</td>
      <td>94.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ashley Yoder</td>
      <td>0.47</td>
      <td>185.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Jinh Yu Frey</td>
      <td>0.38</td>
      <td>236.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>To run through the above code, we first drop the irrelevant columns we won't need. Then we encode inputs of type "object" which are weight_class, stance_f1, and stance_f2. We create a new dataframe out of these encoded values and add them to the cleaned dataframe. After concatenating the df dataframe with a dummy dataframe of "winning_method," we drop "winning_method" as it's no longer needed. The head of this ultimate dataframe is displayed above. Finally, we apply the fit_transform function to the object type columns of the dataframe.</p>
<p>Below we see the correlation factors of every header from our dataframe.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[38]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">15</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">()[[</span><span class="s1">&#39;winner&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s1">&#39;winner&#39;</span><span class="p">,</span> <span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">),</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Correlation Factors of Dataframe Field&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Correlation Factor&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Dataframe Field&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
class="
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Communication-of-Insights-Attained">Communication of Insights Attained<a class="anchor-link" href="#Communication-of-Insights-Attained">&#182;</a></h3>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>As we can see, the only value that isn't in the deep purple color is "winner," which is obviously 1 because every winner won their respective fight.</p>
<p>Since every correlation value is less than or equal to abs(0.031) on a scale of -1 to 1, it's safe to say that we fail to reject the null hypothesis in that we don't have enough evidence to assert that any of these fields has a correlation to winning fights.</p>
<p>As one would expect, the reach, weight, and height qualities are found at the top, but what's most interesting is that the field with the highest correlation factor is Fighter 1's average takedowns per 15 minutes. Apparently the amount of successful takedowns you get in a bout helps you win more than anything else. This coincides with the information on fight ouctomes we previously looked at. Unanimous decisions are the most frequent outcome of fights, and this includes going to the judges for scorecards. This might mean that judges are swayed more by successful takedowns than any other measured metric (judges may be swayed more by something like visual damage/blood/cuts but this isn't something that can be quanitified).</p>
<p>Funnily enough, Fighter 1's stance affects the outcome of a fight the least, which is consistent with my earlier assertion that fighter's stanc is somewhat irrelevant.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">&#182;</a></h1>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Although none of these fields seem to correlate much with winning fights on the gross spectrum across the UFC, this doesn't rule out the possibility that there can be correlation factors if we take a deeper dive into individual fighters. For example, if one wants to predict the outcome of a single fight, they can replicate some of these data analyzing steps with the statistics of the two specific fighters involved in the bout, which will probably give much more skewed results and may give you a statistical advantage in predicting winners.</p>
<p>Sports is an extremely broad category, so if you're not as much interested in UFC or MMA, this tutorial can be applied to other sports you may be interested in. If you're not interested in sports at all, you can still apply this to other games like videogames, chess, etc.</p>
<p>I hope this helped you experience a little taste of working in data science.</p>

</div>
</div>
</div>
</div>
</body>







</html>