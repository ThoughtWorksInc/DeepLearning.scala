/*
DeepLearning.scala by ThoughtWorks
Released for free under the Apache 2.0 license (https://github.com/ThoughtWorksInc/DeepLearning.scala/blob/1.0.x/LICENSE)
*/
function init_init(prefix) {
	return skel.init({
		prefix: prefix,
		resetCSS: true,
		boxModel: 'border',
		grid: {
			gutters: 50
		},
		breakpoints: {
			'mobile': {
				range: '-480',
				lockViewport: true,
				containers: 'fluid',
				grid: {
					collapse: true,
					gutters: 10
				}
			},
			'desktop': {
				range: '481-',
				containers: 1200
			},
			'1000px': {
				range: '481-1200',
				containers: 960
			}
		}
	}, {
		panels: {
			panels: {
				navPanel: {
					breakpoints: 'mobile',
					position: 'left',
					style: 'reveal',
					size: '80%',
					html: '<div data-action="navList" data-args="nav"></div>'
				}
			},
			overlays: {
				titleBar: {
					breakpoints: 'mobile',
					position: 'top-left',
					height: 44,
					width: '100%',
					html: '<span class="toggle" data-action="togglePanel" data-args="navPanel"></span>' +
					'<span class="title" data-action="copyHTML" data-args="logo"></span>'
				}
			}
		}


	});
}
