import org.apache.commons.lang3.SystemUtils

incOptions := incOptions.value.withNameHashing(true).withRecompileOnMacroDef(false)

libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.2.4"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0-M15" % Test

scalaVersion := "2.10.6"

classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" %% "nd4s" % "0.4-rc3.8"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.4-rc3.9"

def osClassifier(moduleId: ModuleID) = {
  if (SystemUtils.IS_OS_MAC_OSX) {
    moduleId classifier s"macosx-${SystemUtils.OS_ARCH}"
  } else if (SystemUtils.IS_OS_LINUX) {
    moduleId classifier s"linux-${SystemUtils.OS_ARCH}"
  } else if (SystemUtils.IS_OS_WINDOWS) {
    moduleId classifier s"windows-${SystemUtils.OS_ARCH}"
  } else {
    moduleId
  }
}

libraryDependencies += osClassifier("org.nd4j" % "nd4j-native" % "0.4-rc3.9" % Test classifier "")

libraryDependencies += "org.scalaz" %%% "scalaz-scalacheck-binding" % "7.2.4"

libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.1"

libraryDependencies ++= {
  if (scalaVersion.value.startsWith("2.10.")) {
    Seq(compilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full))
  } else {
    Seq()
  }
}

scalacOptions in Compile in doc += "-implicits"

scalacOptions in Compile in doc ++= {
  if (scalaBinaryVersion.value == "2.11") {
    Seq("-author")
  } else {
    Seq()
  }
}

addCompilerPlugin("com.milessabin" % "si2712fix-plugin" % "1.2.0" cross CrossVersion.full)

libraryDependencies += "com.thoughtworks.sde" %% "gen" % "2.0.0"

addCompilerPlugin("org.spire-math" % "kind-projector" % "0.8.0" cross CrossVersion.binary)
