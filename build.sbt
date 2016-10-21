lazy val `deep-learning` = project.dependsOn(differentiable)

lazy val differentiable = project.disablePlugins(SparkPackagePlugin)
