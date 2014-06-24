organization := "com.qifun"

name := "stateless-future-util"

version := "0.4.1-SNAPSHOT"

libraryDependencies += "com.qifun" %% "stateless-future" % "0.2.2"

libraryDependencies += "com.dongxiguo" %% "fastring" % "0.2.4"

libraryDependencies += "com.dongxiguo" %% "zero-log" % "0.3.5"

libraryDependencies += "com.dongxiguo.zero-log" %% "context" % "0.3.5"

libraryDependencies += "com.novocode" % "junit-interface" % "0.10" % "test"

scalacOptions ++= Seq("-optimize", "-unchecked", "-Xlint", "-feature")

scalacOptions <++= (scalaVersion) map { sv =>
  if (sv.startsWith("2.10.")) {
    Seq("-deprecation") // Fully compatible with 2.10.x 
  } else {
    Seq() // May use deprecated API in 2.11.x
  }
}

crossScalaVersions := Seq("2.10.4", "2.11.1")

incOptions := incOptions.value.withNameHashing(true)

description := "Utilities for working with Stateless Future."

homepage := Some(url("https://github.com/Atry/stateless-future-util"))

startYear := Some(2014)

licenses := Seq("Apache License, Version 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"))

publishTo <<= (isSnapshot) { isSnapshot: Boolean =>
  if (isSnapshot)
    Some("snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")
  else
    Some("releases" at "https://oss.sonatype.org/service/local/staging/deploy/maven2")
}

scmInfo := Some(ScmInfo(
  url("https://github.com/Atry/stateless-future-util"),
  "scm:git:git://github.com/Atry/stateless-future-util.git",
  Some("scm:git:git@github.com:Atry/stateless-future-util.git")))

pomExtra :=
  <developers>
    <developer>
      <id>Atry</id>
      <name>杨博</name>
      <timezone>+8</timezone>
      <email>pop.atry@gmail.com</email>
    </developer>
  </developers>
