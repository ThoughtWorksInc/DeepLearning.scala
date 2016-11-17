libraryDependencies += "org.nd4j" %% "nd4s" % "0.4-rc3.8"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.4-rc3.9"

SbtNd4J.addNd4jRuntime(Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
