name := "differentiable"

crossScalaVersions := Seq("2.10.6", "2.11.8")

import org.apache.commons.lang3.SystemUtils

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" %% "nd4s" % "0.4-rc3.8"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.4-rc3.9"

def osClassifier(moduleId: ModuleID) = {
  val arch = SystemUtils.OS_ARCH match {
    case "amd64" => "x86_64"
    case other => other
  }
  if (SystemUtils.IS_OS_MAC_OSX) {
    moduleId classifier s"macosx-${arch}"
  } else if (SystemUtils.IS_OS_LINUX) {
    moduleId classifier s"linux-${arch}"
  } else if (SystemUtils.IS_OS_WINDOWS) {
    moduleId classifier s"windows-${arch}"
  } else {
    moduleId
  }
}

libraryDependencies += osClassifier("org.nd4j" % "nd4j-native" % "0.4-rc3.9" % Test classifier "")

libraryDependencies += "org.typelevel" %% "cats" % "0.7.2"

libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("com.milessabin" % "si2712fix-plugin" % "1.2.0" cross CrossVersion.full)

addCompilerPlugin("org.spire-math" % "kind-projector" % "0.8.2" cross CrossVersion.binary)
