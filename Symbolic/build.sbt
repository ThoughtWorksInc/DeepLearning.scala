libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2"

libraryDependencies += "com.jsuereth" %% "scala-arm" % "2.0"

libraryDependencies += "com.github.mpilquist" %% "simulacrum" % "0.10.0"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

description := "Implicit conversion functions that convert from native types to differentiable layers and tapees."
