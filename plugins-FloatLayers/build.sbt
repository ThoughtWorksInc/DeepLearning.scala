incOptions in Test := incOptions.value.withRecompileOnMacroDef(java.util.Optional.of(true))

enablePlugins(Example)

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.1.0" % Test