libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "com.dongxiguo" %% "fastring" % "0.3.1"

libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2"

val lwjglNatives: String = {
  if (util.Properties.isMac) {
    "natives-macos"
  } else if (util.Properties.osName.startsWith("Linux")) {
    "natives-linux"
  } else if (util.Properties.isWin) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(s"lwjgl does not support ${util.Properties.osName}")
  }
}

libraryDependencies += "org.lwjgl" % "lwjgl-opencl" % "3.1.1" % Test

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1" % Test

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1" % Test /* Runtime */ classifier lwjglNatives

fork := true