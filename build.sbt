sbt.dsl.dependsOn(`dynamic-cast`, boolean, double, array2D, hlist, coproduct, seqProject)

lazy val deeplearning = project.disablePlugins(SparkPackagePlugin)

lazy val boolean = project.disablePlugins(SparkPackagePlugin).dependsOn(deeplearning, `buffered-layer`, dslProject)

lazy val double = project.disablePlugins(SparkPackagePlugin).dependsOn(dslProject, boolean, `buffered-layer`)

lazy val `dynamic-cast` = project.disablePlugins(SparkPackagePlugin).dependsOn(dslProject)

lazy val dslProject =
  Project(id = "dsl", base = file("dsl"), dependencies = Seq(deeplearning)).disablePlugins(SparkPackagePlugin)

lazy val seqProject =
  Project(id = "seq", base = file("seq"), dependencies = Seq(dslProject)).disablePlugins(SparkPackagePlugin)

lazy val array2D = project.disablePlugins(SparkPackagePlugin).dependsOn(double)

lazy val hlist = project.disablePlugins(SparkPackagePlugin).dependsOn(dslProject)

lazy val coproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(boolean)

lazy val `buffered-layer` = project.disablePlugins(SparkPackagePlugin).dependsOn(deeplearning)

lazy val `sbt-nd4j` = project

SbtNd4J.addNd4jRuntime(Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

crossScalaVersions := Seq("2.10.6", "2.11.8")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full))

organization in ThisBuild := "com.thoughtworks.deeplearning"
