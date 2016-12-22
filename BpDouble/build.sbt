addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("com.milessabin" % "si2712fix-plugin" % "1.2.0" cross CrossVersion.full)

libraryDependencies += "org.typelevel" %% "cats" % "0.8.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test
