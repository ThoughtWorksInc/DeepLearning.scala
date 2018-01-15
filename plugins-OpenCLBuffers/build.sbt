libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.3" % Test

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

libraryDependencies += "com.thoughtworks.future" %% "future" % "2.0.0-M2"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

scalacOptions += "-Ypartial-unification"

val lwjglNatives: String = {
  import scala.util.Properties._
  if (isMac) {
    "natives-macos"
  } else if (isLinux) {
    "natives-linux"
  } else if (isWin) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(s"lwjgl does not support $osName")
  }
}

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.2"

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.2" % Test classifier lwjglNatives

fork := true
