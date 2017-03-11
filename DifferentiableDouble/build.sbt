addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.typelevel" %% "cats" % "0.8.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "2.0.0" % Test)
