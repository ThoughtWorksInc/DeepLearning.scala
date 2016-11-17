lazy val `deep-learning` = (project in file(".")).dependsOn(differentiable, boolean, double, array2D, hlist, coproduct)

lazy val differentiable = project.disablePlugins(SparkPackagePlugin)

lazy val boolean = project.disablePlugins(SparkPackagePlugin).dependsOn(any)

lazy val double = project.disablePlugins(SparkPackagePlugin).dependsOn(any, boolean)

lazy val any = project.disablePlugins(SparkPackagePlugin).dependsOn(differentiable)

lazy val array2D = project.disablePlugins(SparkPackagePlugin).dependsOn(any, double)

lazy val hlist = project.disablePlugins(SparkPackagePlugin).dependsOn(any)

lazy val coproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(any)
