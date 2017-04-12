libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

libraryDependencies += "org.scalaz" %% "scalaz-effect" % "7.2.10" % Test

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
