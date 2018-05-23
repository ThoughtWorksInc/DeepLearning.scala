libraryDependencies += "com.thoughtworks.dsl" %% "dsl" % "1.0.0-RC9"

libraryDependencies += "com.thoughtworks.dsl" %% "domains-scalaz" % "1.0.0-RC9" % Test

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "1.0.0-RC9")

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-reseteverywhere" % "1.0.0-RC9")

fork in Test := true

enablePlugins(Example)
