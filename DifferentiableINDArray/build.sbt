libraryDependencies += "org.nd4j" %% "nd4s" % "0.7.1"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.7.1"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.7.1"

fork in Test := true
