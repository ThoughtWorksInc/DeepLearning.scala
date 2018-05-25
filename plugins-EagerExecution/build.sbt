libraryDependencies += "com.thoughtworks.dsl" %% "dsl" % "1.0.0-RC10"

libraryDependencies += "com.thoughtworks.dsl" %% "domains-scalaz" % "1.0.0-RC10" % Test

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "1.0.0-RC10")

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-reseteverywhere" % "1.0.0-RC10")

fork in Test := true

enablePlugins(Example)
