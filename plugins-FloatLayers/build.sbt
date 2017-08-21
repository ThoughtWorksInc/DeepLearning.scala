incOptions in Test := incOptions.value.withRecompileOnMacroDef(true)

enablePlugins(Example)

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.1.0-M0" % Test