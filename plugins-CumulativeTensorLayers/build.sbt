libraryDependencies += ("org.lwjgl" % "lwjgl" % "3.1.6" % Optional).jar().classifier {
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

fork := true

enablePlugins(Example)

import scala.meta._
exampleSuperTypes := exampleSuperTypes.value.map {
  case ctor"_root_.org.scalatest.FreeSpec" =>
    ctor"_root_.org.scalatest.AsyncFreeSpec"
  case otherTrait =>
    otherTrait
}

scalacOptions += "-Ypartial-unification"

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.4" % Test
